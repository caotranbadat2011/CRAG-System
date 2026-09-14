# CRAG System — ingestion and corrective evidence workflow

This repository implements the auditable ingestion layer for a Corrective RAG system:

```text
source file -> immutable raw copy -> parse -> clean -> structural chunks
            -> BGE-M3 dense + lexical weights -> Qdrant hybrid index
            -> validation/search smoke test
```

## Supported input

- PDF (`.pdf`), including layout-aware text, headings, lists, tables,
  formulas, columns, fonts, coordinates, extracted images, and passwords
- Word (`.docx`), including rich text, localized/custom headings, lists,
  nested tables, text boxes, headers/footers, footnotes, comments, and images
- Markdown (`.md`, `.markdown`), including CommonMark/GFM structure, clean
  inline text, source ranges, references, local image validation, and OCR hooks
- Plain text (`.txt`) with safe multi-encoding fallback

Unsupported files are rejected explicitly. Empty documents, oversized inputs, unreadable PDFs, malformed DOCX archives, zero-token chunks, and inconsistent vector dimensions fail before indexing.

## Project layout

```text
src/crag_ingestion/
  parsers/       format-specific extraction and parser registry
  embeddings/    BGE-M3 dense/sparse adapter and embedding contract
  index/         Qdrant named vectors, payload, filtering, and hybrid search
  retrieval/     BGE cross-encoder reranking and post-branch semantic diversity
                 Gemini evaluation, internal refinement, query rewrite, DDGS web
                 search, bounded page extraction, web filtering, context assembly,
                 Gemini answer generation and citation validation
  cleaning.py    Unicode/whitespace/control-char/margin normalization
  chunking.py    structure-aware chunking with overlap and provenance
  storage.py     atomic raw and processed artifact persistence
  documents.py   upload, update, refresh, and delete document lifecycle
  pipeline.py    ingestion and retrieval entry points
  workflow.py    three-branch LangGraph routing and SQLite checkpoints
  web.py         local JSON API and browser app server
  static/        citation-aware browser UI
  validation.py  integrity and traceability checks
  cli.py         operational interface
tests/           unit, format parser, index, and end-to-end tests
data/            runtime artifacts (ignored by Git)
```

Runtime artifacts are separated:

```text
data/raw/<document-id>/<sha256>.<ext>   immutable source version
data/raw/<document-id>/<sha256>.media/  content-addressed DOCX/PDF images
data/processed/<document-id>/<sha256>.json  inspectable blocks and chunks
data/qdrant/                            embedded Qdrant storage for local runs
data/checkpoints/crag.sqlite3           LangGraph run history and evidence state
data/uploads/<upload-id>/<filename>      browser-managed source files
```

## Setup and usage

```powershell
python -m pip install -e ".[dev]"
python -m crag_ingestion ingest .\documents
python -m crag_ingestion list
python -m crag_ingestion query "chính sách hoàn tiền" --limit 5
python -m crag_ingestion query "chính sách hoàn tiền" --rerank --candidate-limit 30 --limit 10
python -m crag_ingestion evaluate "chính sách hoàn tiền" --candidate-limit 30 --limit 10
python -m crag_ingestion refine "chính sách hoàn tiền" --candidate-limit 30 --limit 10
python -m crag_ingestion search-web "chính sách hoàn tiền" --candidate-limit 30 --limit 10
python -m crag_ingestion run "chính sách hoàn tiền" --candidate-limit 30 --limit 10
python -m crag_ingestion show-run RUN_ID
python -m crag_ingestion serve --port 8000
python -m crag_ingestion validate
pytest
```

Open `http://127.0.0.1:8000/` after starting `serve`. The local browser app
asks questions, displays the answer and branch, and lets you click each
`[n]` citation to inspect the exact selected strip. Internal citations open
the immutable raw source version captured by that run (PDF links include a
page anchor); web citations open the fetched page URL. If a source was later
deleted, its text remains in the checkpoint but the original file link may
no longer be available.

The document panel supports uploading a new source, replacing a browser-uploaded
source with a file of the same extension, refreshing any indexed document from
its existing source path, and removing a document from Qdrant. Removal also
deletes CRAG-managed raw/processed copies and browser-uploaded source files;
it never deletes an external source file you indexed through the CLI. Historical
SQLite checkpoints are retained. Uploads are limited by `max_file_bytes`
(50 MiB by default) and supported parser extensions.

The JSON API has `POST /api/ask`, `GET /api/runs/{run_id}`,
`GET/POST /api/documents`, `GET/PUT/DELETE /api/documents/{document_id}`,
`POST /api/documents/{document_id}/refresh`, and read-only source endpoints.
Mutating requests require JSON and `X-CRAG-Local: 1`. The unauthenticated
server binds to loopback only; do not expose it through a reverse proxy or
port forwarding without adding authentication and authorization. Like the
CLI, it sends selected evidence to Gemini and stores run evidence in SQLite.

Without `--qdrant-url`, the CLI uses Qdrant local mode under `data/qdrant`.
For a Qdrant server, pass its URL and optionally keep the API key in a file:

```powershell
python -m crag_ingestion `
  --qdrant-url http://localhost:6333 `
  --qdrant-collection crag_bge_m3_hybrid `
  ingest .\documents

python -m crag_ingestion `
  --qdrant-url https://your-cluster.example `
  --qdrant-api-key-file .\secrets\qdrant-api-key.txt `
  query "chính sách hoàn tiền"
```

Every chunk is stored as a Qdrant point. The point payload retains document and
chunk identifiers, source and artifact paths, content and pipeline hashes, model,
text, heading path, pages, parser metadata, and provenance. Collections use
named `dense` (1024 dimensions, Cosine) and `sparse` vectors.

FlagEmbedding's BGE-M3 adapter is the only embedding backend. A single model
inference produces the normalized 1024-dimensional `dense_vecs` and token-ID
`lexical_weights`; the latter are stored as Qdrant sparse vectors. The default
model is `BAAI/bge-m3`. `--embedding-model` accepts a BGE-M3-compatible model
or local directory, not an arbitrary Sentence Transformers model. Querying
prefetches dense and sparse candidates with the same filters and merges them
with Qdrant reciprocal rank fusion (RRF).

`query --rerank` takes up to `--candidate-limit` hybrid/RRF results, scores each
query–chunk pair with `BAAI/bge-reranker-v2-m3`, and returns the best `--limit`
chunks for a future retrieval evaluator. Its output keeps both
`retrieval_score` (RRF) and `rerank_score` (cross-encoder logit); neither is a
calibrated confidence for the Correct/Incorrect/Ambiguous decision. The model
is loaded lazily and can be changed with `--reranker-model`. Python callers can
use `IngestionPipeline.retrieve_for_evaluation(...)` directly. This stage
reorders retrieved chunks; it does not change stored Qdrant vectors.

`SemanticDiversityFilter` is deliberately separate from evaluator input. After
the CRAG branch has refined internal and/or web knowledge into `KnowledgeStrip`
objects, call `IngestionPipeline.select_diverse_context(...)` to select
nonredundant passages by BGE-M3 dense-vector MMR. Each strip retains its
`source_ref` (chunk ID or URL) for citations. For the Ambiguous branch, pass
`min_per_source={"internal": 1, "web": 1}` to require both sources; an
unsatisfiable quota raises an error instead of silently dropping a source.
Internal refinement, web search, and cited context assembly are available.
`run` also generates a cited answer from the selected context.

The `evaluate` command runs hybrid retrieval, BGE reranking, then a single
Gemini `generateContent` request for per-chunk labels. It defaults to the
stable `gemini-3.5-flash-lite` model, which Google currently lists with a free tier.
The request includes a structured JSON schema, and the response is also
validated locally for every chunk ID,
label, and explanation before the decision is accepted. At least one relevant
chunk selects `Correct`; all irrelevant selects `Incorrect`; otherwise it
selects `Ambiguous`. A missing key, rate limit, incomplete response, or invalid
JSON fails explicitly rather than silently selecting a branch. These labels
are not calibrated probabilities or a reproduction of the paper's fine-tuned
T5 evaluator and should be checked against a labeled evaluation set.

`refine` first evaluates the reranked chunks, then decomposes only internal
evidence from `Correct` (relevant parents) or `Ambiguous` (relevant/uncertain
parents) into short, source-ordered strips. It makes additional batched Gemini
judgments for the strips and keeps only those labeled relevant. `Incorrect`
returns no internal strips. Defaults are 360 characters, at most two sentences
per strip, and 12 strips per evaluator call; the global flags
`--strip-max-chars`, `--strip-max-sentences`, and `--strip-batch-size` tune these
without re-ingesting. Each `KnowledgeStrip` retains its parent `chunk_id`,
document ID, source/raw paths, pages, headings, parser source metadata, and
exact half-open character offsets **within the indexed chunk**. Parser source
ranges are preserved but cannot always be narrowed to a strip because the
chunker may combine blocks or prepend overlap. The returned strips are ready
for post-branch semantic diversity selection; `refine` does not call that
selector or generate an answer.

`search-web` evaluates retrieved chunks first. `Correct` makes no web request;
`Incorrect` and `Ambiguous` use Gemini to rewrite up to two short queries,
search DuckDuckGo through `ddgs` with `backend="duckduckgo"`, fetch at most five
HTML/plain-text pages, split page text into short passages, rank them locally
with BGE reranker, and keep only passages Gemini labels relevant. Search snippets
are never treated as evidence. The JSON result includes each web strip's URL,
page title, query, fetch timestamp, content hash, and half-open offsets in the
extracted page text. For `Ambiguous`, call both `refine_internal_knowledge` and
`search_web_knowledge`, then apply `select_diverse_context` with optional
source quotas. For `Incorrect`, only the web strips should feed that selector.
This command prepares cited evidence but does not synthesize an answer; use
`run` for the complete workflow.
DDGS results with blank or nonpublic `href` values are discarded before page
selection. A search with no usable URL or a transient DDGS error is retried
up to three total attempts with 0.5s/1s backoff by default. If all rewritten
queries still yield no eligible URL, the original question is tried once with
the same bounded retry policy (unless it duplicates a rewritten query).
`queries` in the result lists the queries actually attempted. Warnings report
the query number, attempt count, invalid/empty URL counts, and exception *type*
without echoing raw exception messages or rejected URLs.

`run` executes the complete corrective evidence workflow with LangGraph:

```text
hybrid retrieve → BGE rerank → Gemini evaluate
  Correct   → internal refinement ────────────────┐
  Incorrect → web knowledge search ───────────────┼→ diversity filter → cited context
  Ambiguous → internal refinement → web search ───┘                     → Gemini answer
```

The Ambiguous route asks the diversity selector to retain at least one strip
from each source when both have evidence. If a source is missing, the strip
limit is too small, or sources are near-duplicates, the result is marked
`partial` with a warning. If no relevant strip remains, the status is
`no_evidence`; no search snippet is substituted. `ready` means selected
evidence exists (and both sources are represented for Ambiguous), not that
the evidence has been independently fact-checked.

The output contains a bounded JSONL `context_text` with `[1]`, `[2]`, …
markers, matching `citations` containing the original chunk ID or URL and
source offsets/metadata. Strips are never cut mid-text to fit the context
budget; over-budget strips are omitted with a warning. Use global
`--context-max-strips` and `--context-max-chars` before `run` to tune the
default 8 strips and 6,000 characters. The diversity weights remain 0.6
for relevance and 0.85 for duplicate rejection.

The answer node uses the configured `--evaluator-model` (default
`gemini-3.5-flash-lite`) and the existing `GEMINI_API_KEY`. Gemini returns
atomic claims with citation-marker arrays; the application validates every
marker against the selected context and attaches the matching source metadata
in `answer.citations`. Invalid or missing markers fail explicitly. For
`partial` context, the answer starts with an evidence-limitation notice. For
`no_evidence`, no generation API call is made and the answer abstains. Gemini
may also abstain when available evidence does not answer the question. Citation
validation checks references, not whether each claim is actually supported;
important claims still need independent factual review.
The answer defaults to at most 12 claims and 4,096 Gemini output tokens;
`--answer-max-claims` (1–20) and `--answer-max-output-tokens` (512–8,192)
can be passed before `run` to adjust detail. More tokens cannot create
facts absent from the selected context.

Each `run` creates a new SQLite checkpoint thread and returns its `run_id`.
`show-run RUN_ID` reads its completed result without reopening Qdrant or
calling models/APIs. The database is under `data/checkpoints/` by default and
is ignored by Git. It stores retrieved text and web evidence, so protect it
as sensitive data and apply a retention policy for longer-running deployments.
Checkpoint deserialization uses LangGraph's strict msgpack allowlist. This
local SQLite saver is intended for lightweight synchronous use, not a
multi-worker production service. New runs checkpoint the answer as well as the
context. Older context-only checkpoints remain readable with `answer: null`.

Global options `--web-max-pages`, `--web-results-per-query`, `--web-region`,
`--web-search-attempts`, `--web-search-backoff`, and repeatable
`--web-allowed-domain` limit web retrieval; options precede the
subcommand. For example:

```powershell
python -m crag_ingestion --web-max-pages 3 `
  --web-allowed-domain example.org search-web "chính sách hoàn tiền"
```

Page fetching accepts only public HTTP(S) targets on standard ports, checks
DNS and redirects, rejects oversized or non-text responses, and removes common
navigation/script content. These checks reduce risk but are not a sandbox for
hostile pages. Retrieved web content and questions are sent to Gemini for
relevance judgment; review provider privacy requirements and factual claims
before using them in an answer. DDGS availability and site access can vary,
so failed searches/pages are reported as warnings. A page with no relevant
strip yields no external evidence rather than falling back to its snippet.

Create a Gemini API key in Google AI Studio. Never store it in the repository;
set `GEMINI_API_KEY` only in your current process:

```powershell
$secret = Read-Host "Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
python -m crag_ingestion evaluate "chính sách hoàn tiền" --limit 5
Remove-Item Env:GEMINI_API_KEY
```

The evaluator sends retrieved chunk text to Gemini. Do not enable it for
confidential documents without checking your data-sharing requirements and
the chosen provider's policies. Google's [pricing page](https://ai.google.dev/gemini-api/docs/pricing)
currently says free-tier data may be used to improve its products. The embedding
and reranker remain local. An environment variable already set in PowerShell
takes precedence over `.env`. To use another Gemini model, put the global
option before the command, e.g. `python -m crag_ingestion --evaluator-model gemini-3.5-flash-lite evaluate "câu hỏi"`.
HTTP 429 indicates quota/rate limits; HTTP 404 usually means the selected model
is unavailable. Provider error messages are not echoed because they may contain
document text or sensitive data.

The default Qdrant collection is `crag_bge_m3_hybrid`. The prior dense-only
`crag_bge_m3` collection is left untouched; re-ingest source documents into the
new collection. An explicitly selected collection with an incompatible vector
schema or dimension is rejected rather than overwritten.

Keep the same model for ingestion and querying. A change to cleaning, chunking,
parser support, model, or model-derived dimensions changes the pipeline signature;
re-ingestion then replaces the document atomically. Unchanged content is skipped
by default, while `--force` rebuilds it.

Retrieval checks the stored pipeline signature and original source checksum
before using any indexed chunk. When a parser or other ingestion setting has
changed, the source file has changed, or the source is missing, `query`,
`evaluate`, and `run` stop with an actionable error instead of sending stale
evidence to the evaluator or answer model. `list` and the web document list
show `index_status` (`current`, `outdated_pipeline`, `source_changed`, or
`source_missing`). Use **Làm mới** in the local web app or re-ingest the
original file to rebuild an outdated document. A missing original source must
be restored or uploaded again; old saved runs remain historical snapshots.

## Validation contract

`validate` checks Qdrant collection status, dense/sparse schema, distance and
dimensions, point and
document chunk counts, raw/processed artifact existence, contiguous ordinals,
text lengths, payload structure, dense dimensions and L2 norms, sparse token IDs
and lexical weights, provenance,
and PDF page references. It exits non-zero if any error is found; warnings remain
visible without failing the index.

DOCX formatting is stored as ordered run metadata on each block rather than mixed
into embedding text. List definitions, table hierarchy, note/comment identifiers,
part names, image hashes, MIME types, alternative text, and extracted asset paths
remain available in processed artifacts and chunk source metadata.

Markdown parsing follows CommonMark plus GFM tables, strikethrough, task lists,
footnotes, definition lists, YAML front matter, and colon-fence admonitions. It
emits clean embedding text while preserving inline formatting, links, images,
table/list structure, reference definitions, and exact source line/character
ranges as metadata. Local image targets are validated, and callers may inject an
OCR function through `MarkdownParser(image_ocr=...)` without coupling ingestion
to one OCR engine.

PDF parsing uses `pdfplumber` for word coordinates, reading order, multi-column
layout, and tables, with a coordinate-aware `pypdf` fallback. Font name, size,
bold/italic flags, bounding boxes, list markers, formula hints, headings, vector
graphics, and page provenance are retained. Embedded images are extracted by
content hash. Image-only pages retain image blocks and emit a warning when they
have no extractable text; no image text recognition is performed. Encrypted files
accept a direct password, a per-file password provider, or the CLI's
`--pdf-password-file` option.
