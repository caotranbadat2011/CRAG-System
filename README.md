# CRAG System — document ingestion foundation

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
  cleaning.py    Unicode/whitespace/control-char/margin normalization
  chunking.py    structure-aware chunking with overlap and provenance
  storage.py     atomic raw and processed artifact persistence
  pipeline.py    idempotent orchestration
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
```

## Setup and usage

```powershell
python -m pip install -e ".[dev]"
python -m crag_ingestion ingest .\documents
python -m crag_ingestion list
python -m crag_ingestion query "chính sách hoàn tiền" --limit 5
python -m crag_ingestion query "chính sách hoàn tiền" --rerank --candidate-limit 30 --limit 10
python -m crag_ingestion validate
pytest
```

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
The evaluator, refinement, web search, and context assembly are not yet built.

The default Qdrant collection is `crag_bge_m3_hybrid`. The prior dense-only
`crag_bge_m3` collection is left untouched; re-ingest source documents into the
new collection. An explicitly selected collection with an incompatible vector
schema or dimension is rejected rather than overwritten.

Keep the same model for ingestion and querying. A change to cleaning, chunking,
parser support, model, or model-derived dimensions changes the pipeline signature;
re-ingestion then replaces the document atomically. Unchanged content is skipped
by default, while `--force` rebuilds it.

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
