# CRAG System — document ingestion foundation

This repository implements the auditable ingestion layer for a Corrective RAG system:

```text
source file -> immutable raw copy -> parse -> clean -> structural chunks
            -> embeddings -> SQLite vector index -> validation/search smoke test
```

## Supported input

- PDF (`.pdf`), including page-level provenance
- Word (`.docx`), including rich text, localized/custom headings, lists,
  nested tables, text boxes, headers/footers, footnotes, comments, and images
- Markdown (`.md`, `.markdown`), including heading paths
- Plain text (`.txt`) with safe multi-encoding fallback

Unsupported files are rejected explicitly. Empty documents, oversized inputs, unreadable PDFs, malformed DOCX archives, zero-token chunks, and inconsistent vector dimensions fail before indexing.

## Project layout

```text
src/crag_ingestion/
  parsers/       format-specific extraction and parser registry
  embeddings/    provider contract, offline baseline, semantic adapter
  index/         transactional SQLite vector index
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
data/raw/<document-id>/<sha256>.media/  content-addressed DOCX images
data/processed/<document-id>/<sha256>.json  inspectable blocks and chunks
data/index/crag.sqlite3                 document, provenance, vector index
```

## Setup and usage

```powershell
python -m pip install -e ".[dev]"
python -m crag_ingestion ingest .\documents
python -m crag_ingestion list
python -m crag_ingestion query "chính sách hoàn tiền" --limit 5
python -m crag_ingestion validate
pytest
```

The default `hash` embedder is deterministic, offline, and intended for reliable ingestion verification. It provides lexical retrieval, not production semantic quality. To use multilingual semantic embeddings:

```powershell
python -m pip install -e ".[semantic]"
python -m crag_ingestion --embedding-provider sentence-transformers ingest .\documents
```

Keep the same embedding provider/model for ingestion and querying. A change to cleaning, chunking, parser support, model, or dimensions changes the pipeline signature; re-ingestion then replaces the document atomically. Unchanged content is skipped by default, while `--force` rebuilds it.

## Validation contract

`validate` checks SQLite integrity and foreign keys, raw/processed artifact existence, declared versus actual chunk counts, contiguous ordinals, text lengths, JSON provenance, embedding dimensions and L2 norms, and PDF page references. It exits non-zero if any error is found; warnings remain visible without failing the index.

DOCX formatting is stored as ordered run metadata on each block rather than mixed
into embedding text. List definitions, table hierarchy, note/comment identifiers,
part names, image hashes, MIME types, alternative text, and extracted asset paths
remain available in processed artifacts and chunk source metadata.
