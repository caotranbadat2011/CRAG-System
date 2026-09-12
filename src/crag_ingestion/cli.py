from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ChunkingConfig, IngestionConfig
from .exceptions import IngestionError
from .pipeline import IngestionPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crag-ingest", description="CRAG document ingestion pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Artifact and index directory")
    parser.add_argument("--embedding-provider", choices=("hash", "sentence-transformers"), default="hash")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--dimensions", type=int, default=384, help="Dimensions for hash embeddings")
    parser.add_argument("--max-chars", type=int, default=1_200)
    parser.add_argument("--overlap-chars", type=int, default=180)
    parser.add_argument("--min-chars", type=int, default=120)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest one file or a directory")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--no-recursive", action="store_true")
    ingest.add_argument("--force", action="store_true")
    query = subparsers.add_parser("query", help="Run a retrieval smoke test")
    query.add_argument("text")
    query.add_argument("--limit", type=int, default=5)
    query.add_argument("--document-id")
    subparsers.add_parser("list", help="List indexed documents")
    validate = subparsers.add_parser("validate", help="Check index and provenance invariants")
    validate.add_argument("--skip-file-checks", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> IngestionConfig:
    return IngestionConfig(
        data_dir=args.data_dir,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.dimensions,
        chunking=ChunkingConfig(args.max_chars, args.overlap_chars, args.min_chars),
    )


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with IngestionPipeline(_config(args)) as pipeline:
            if args.command == "ingest":
                files = pipeline.discover(args.path, recursive=not args.no_recursive)
                results: list[dict[str, object]] = []
                failures: list[dict[str, str]] = []
                for path in files:
                    try:
                        results.append(pipeline.ingest_file(path, force=args.force).to_dict())
                    except (IngestionError, OSError, ValueError) as exc:
                        failures.append({"path": str(path), "error": str(exc)})
                _print({"discovered": len(files), "results": results, "failures": failures})
                return 1 if failures else 0
            if args.command == "query":
                _print([item.to_dict() for item in pipeline.search(args.text, args.limit, args.document_id)])
                return 0
            if args.command == "list":
                _print(pipeline.index.documents())
                return 0
            if args.command == "validate":
                report = pipeline.validate(check_files=not args.skip_file_checks)
                _print(report.to_dict())
                return 0 if report.valid else 1
    except (IngestionError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

