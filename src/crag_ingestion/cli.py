from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    AnswerConfig,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_RERANKER_MODEL,
    ChunkingConfig,
    ContextConfig,
    IngestionConfig,
    RefinementConfig,
    WebSearchConfig,
)
from .exceptions import IngestionError
from .pipeline import IngestionPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crag-ingest", description="CRAG document ingestion pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Artifact and index directory")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="BGE-M3-compatible FlagEmbedding model name or local model path",
    )
    parser.add_argument("--max-chars", type=int, default=1_200)
    parser.add_argument("--overlap-chars", type=int, default=180)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument(
        "--pdf-password-file",
        type=Path,
        help="Read a PDF password from a file instead of exposing it in command history",
    )
    parser.add_argument(
        "--qdrant-url",
        help="Qdrant server URL; omit to use embedded local storage under DATA_DIR/qdrant",
    )
    parser.add_argument(
        "--qdrant-api-key-file",
        type=Path,
        help="Read the Qdrant API key from a file",
    )
    parser.add_argument("--qdrant-collection", default=DEFAULT_QDRANT_COLLECTION)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    parser.add_argument("--strip-max-chars", type=int, default=360)
    parser.add_argument("--strip-max-sentences", type=int, default=2)
    parser.add_argument("--strip-batch-size", type=int, default=12)
    parser.add_argument("--qdrant-timeout", type=int, default=30)
    parser.add_argument("--web-max-pages", type=int, default=5)
    parser.add_argument("--web-results-per-query", type=int, default=5)
    parser.add_argument("--web-region", default="vn-vi")
    parser.add_argument("--web-search-attempts", type=int, default=3)
    parser.add_argument("--web-search-backoff", type=float, default=0.5)
    parser.add_argument("--web-allowed-domain", action="append", default=[])
    parser.add_argument("--context-max-strips", type=int, default=8)
    parser.add_argument("--context-max-chars", type=int, default=6_000)
    parser.add_argument("--answer-max-claims", type=int, default=12)
    parser.add_argument("--answer-max-output-tokens", type=int, default=4_096)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest one file or a directory")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--no-recursive", action="store_true")
    ingest.add_argument("--force", action="store_true")
    query = subparsers.add_parser("query", help="Run a retrieval smoke test")
    query.add_argument("text")
    query.add_argument("--limit", type=int, default=5)
    query.add_argument("--document-id")
    query.add_argument("--rerank", action="store_true", help="Rerank hybrid candidates with BGE")
    query.add_argument("--candidate-limit", type=int, default=30)
    evaluate = subparsers.add_parser("evaluate", help="Rerank and judge retrieval with Gemini")
    evaluate.add_argument("text")
    evaluate.add_argument("--candidate-limit", type=int, default=30)
    evaluate.add_argument("--limit", type=int, default=10)
    evaluate.add_argument("--document-id")
    refine = subparsers.add_parser("refine", help="Evaluate and refine internal knowledge strips")
    refine.add_argument("text")
    refine.add_argument("--candidate-limit", type=int, default=30)
    refine.add_argument("--limit", type=int, default=10)
    refine.add_argument("--document-id")
    search_web = subparsers.add_parser(
        "search-web", help="Evaluate retrieval and search/verify external knowledge"
    )
    search_web.add_argument("text")
    search_web.add_argument("--candidate-limit", type=int, default=30)
    search_web.add_argument("--limit", type=int, default=10)
    search_web.add_argument("--document-id")
    run = subparsers.add_parser("run", help="Run three-branch CRAG and generate a cited answer")
    run.add_argument("text")
    run.add_argument("--candidate-limit", type=int, default=30)
    run.add_argument("--limit", type=int, default=10)
    run.add_argument("--document-id")
    show_run = subparsers.add_parser("show-run", help="Read a completed SQLite checkpoint")
    show_run.add_argument("run_id")
    web = subparsers.add_parser("serve", help="Start the local question-answering web app")
    web.add_argument("--port", type=int, default=8000)
    subparsers.add_parser("list", help="List indexed documents")
    validate = subparsers.add_parser("validate", help="Check index and provenance invariants")
    validate.add_argument("--skip-file-checks", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> IngestionConfig:
    pdf_password = None
    if args.pdf_password_file:
        pdf_password = args.pdf_password_file.read_text(encoding="utf-8").rstrip("\r\n")
    qdrant_api_key = None
    if args.qdrant_api_key_file:
        qdrant_api_key = args.qdrant_api_key_file.read_text(encoding="utf-8").strip()
    return IngestionConfig(
        data_dir=args.data_dir,
        pdf_password=pdf_password,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        evaluator_model=args.evaluator_model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=args.qdrant_collection,
        qdrant_timeout=args.qdrant_timeout,
        chunking=ChunkingConfig(args.max_chars, args.overlap_chars, args.min_chars),
        refinement=RefinementConfig(
            args.strip_max_chars, args.strip_max_sentences, args.strip_batch_size
        ),
        web_search=WebSearchConfig(
            max_pages=args.web_max_pages,
            results_per_query=args.web_results_per_query,
            region=args.web_region,
            search_attempts=args.web_search_attempts,
            search_backoff_seconds=args.web_search_backoff,
            allowed_domains=tuple(args.web_allowed_domain),
        ),
        context=ContextConfig(
            max_strips=args.context_max_strips,
            max_chars=args.context_max_chars,
        ),
        answer=AnswerConfig(args.answer_max_claims, args.answer_max_output_tokens),
    )


def _print(value: object) -> None:
    # Windows terminals (and redirected Codex terminals) may expose an ASCII
    # stream even when the process environment requests UTF-8.  Reconfigure
    # when possible, then fall back to escaped Unicode instead of aborting
    # after the pipeline has completed successfully.
    output = json.dumps(value, ensure_ascii=False, indent=2)
    stream = sys.stdout
    try:
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        print(output, file=stream)
    except (AttributeError, OSError, UnicodeEncodeError):
        print(json.dumps(value, ensure_ascii=True, indent=2), file=stream)


def _print_error(message: str) -> None:
    text = f"error: {message}\n"
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except UnicodeEncodeError:
        raw = getattr(sys.stderr, "buffer", None)
        if raw is not None:
            raw.write(text.encode("utf-8", errors="backslashreplace"))
            raw.flush()
        else:
            sys.stderr.write(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass
    args = _parser().parse_args(argv)
    try:
        if args.command == "serve":
            from .web import serve

            serve(_config(args), port=args.port)
            return 0
        if args.command == "show-run":
            from .workflow import CragWorkflow

            _print(CragWorkflow.load_run(_config(args).checkpoint_path, args.run_id).to_dict())
            return 0
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
                if args.rerank:
                    results = pipeline.retrieve_for_evaluation(
                        args.text,
                        candidate_limit=args.candidate_limit,
                        evaluation_limit=args.limit,
                        document_id=args.document_id,
                    )
                else:
                    results = pipeline.search(args.text, args.limit, args.document_id)
                _print([item.to_dict() for item in results])
                return 0
            if args.command == "evaluate":
                decision = pipeline.evaluate_retrieval(
                    args.text,
                    candidate_limit=args.candidate_limit,
                    evaluation_limit=args.limit,
                    document_id=args.document_id,
                )
                _print(decision.to_dict())
                return 0
            if args.command == "refine":
                decision = pipeline.evaluate_retrieval(
                    args.text,
                    candidate_limit=args.candidate_limit,
                    evaluation_limit=args.limit,
                    document_id=args.document_id,
                )
                strips = pipeline.refine_internal_knowledge(args.text, decision)
                _print({
                    "decision": decision.to_dict(),
                    "internal_strips": [strip.to_dict() for strip in strips],
                })
                return 0
            if args.command == "search-web":
                decision = pipeline.evaluate_retrieval(
                    args.text,
                    candidate_limit=args.candidate_limit,
                    evaluation_limit=args.limit,
                    document_id=args.document_id,
                )
                result = pipeline.search_web_knowledge(args.text, decision)
                _print({"decision": decision.to_dict(), **result.to_dict()})
                return 0
            if args.command == "run":
                result = pipeline.run_crag(
                    args.text,
                    candidate_limit=args.candidate_limit,
                    evaluation_limit=args.limit,
                    document_id=args.document_id,
                )
                _print(result.to_dict())
                return 0
            if args.command == "list":
                from .documents import DocumentManager

                _print(DocumentManager(pipeline).list_documents())
                return 0
            if args.command == "validate":
                report = pipeline.validate(check_files=not args.skip_file_checks)
                _print(report.to_dict())
                return 0 if report.valid else 1
    except (IngestionError, OSError, ValueError, RuntimeError) as exc:
        _print_error(str(exc))
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
