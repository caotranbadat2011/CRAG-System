from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .chunking import StructuralChunker
from .cleaning import DocumentCleaner
from .config import IngestionConfig
from .embeddings import Embedder, EmbeddingVector, create_embedder
from .exceptions import EmptyDocumentError, FileTooLargeError, IngestionError, StaleIndexError
from .index import QdrantVectorIndex
from .models import IngestionResult, SearchResult
from .parsers import ParserRegistry, default_registry
from .retrieval import (
    AnswerGenerator, AssembledContext, BGEReranker, CorrectiveAction,
    DDGSWebSearchProvider, EvaluationDecision, GeneratedAnswer, GeminiAnswerGenerator,
    GeminiQueryRewriter, GeminiRetrievalEvaluator, KnowledgeRefiner, KnowledgeStrip,
    PageFetcher, QueryRewriter, RerankedHit, RerankedRetriever, Reranker,
    RetrievalEvaluator, SafePageFetcher, SemanticDiversityFilter, WebKnowledgeResult,
    WebKnowledgeSearcher, WebSearchProvider,
)
from .storage import ArtifactStore
from .utils import canonical_json, sha256_file, stable_document_id
from .validation import ValidationReport, validate_index

if TYPE_CHECKING:
    from .workflow import CragRunResult


class IngestionPipeline:
    # Bump whenever parsing semantics change so existing documents are rebuilt.
    PIPELINE_VERSION = "9"

    def __init__(
        self,
        config: IngestionConfig | None = None,
        *,
        registry: ParserRegistry | None = None,
        embedder: Embedder | None = None,
        index: QdrantVectorIndex | None = None,
        reranker: Reranker | None = None,
        evaluator: RetrievalEvaluator | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.config = config or IngestionConfig()
        self.registry = registry or default_registry(self.config)
        self.embedder = embedder or create_embedder(self.config)
        self._reranker = reranker
        self._evaluator = evaluator
        self._answer_generator = answer_generator
        self.cleaner = DocumentCleaner(self.config.cleaning)
        self.chunker = StructuralChunker(self.config.chunking)
        self.artifacts = ArtifactStore(self.config.raw_dir, self.config.processed_dir)
        self.index = index or QdrantVectorIndex(
            path=self.config.qdrant_path,
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            collection_name=self.config.qdrant_collection,
            timeout=self.config.qdrant_timeout,
        )
        self._owns_index = index is None

    @property
    def pipeline_signature(self) -> str:
        settings = {
            "version": self.PIPELINE_VERSION,
            "cleaning": asdict(self.config.cleaning),
            "chunking": asdict(self.config.chunking),
            "embedder": self.embedder.name,
            "dimensions": self.embedder.dimensions,
            "parsers": self.registry.parser_signatures,
        }
        return hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()

    def close(self) -> None:
        if self._owns_index:
            self.index.close()

    def __enter__(self) -> "IngestionPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def ingest_file(self, path: str | Path, force: bool = False) -> IngestionResult:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise IngestionError(f"Input is not a file: {source}")
        parser = self.registry.parser_for(source)
        size = source.stat().st_size
        if size > self.config.max_file_bytes:
            raise FileTooLargeError(
                f"File is {size} bytes; limit is {self.config.max_file_bytes}: {source}"
            )

        document_id = stable_document_id(source)
        content_hash = sha256_file(source)
        signature = self.pipeline_signature
        if not force and self.index.is_current(document_id, content_hash, signature):
            existing = self.index.get_document(document_id)
            assert existing is not None
            return IngestionResult(
                document_id=document_id,
                source_path=str(source),
                stored_path=existing["stored_path"],
                content_hash=content_hash,
                status="skipped",
                block_count=existing["block_count"],
                chunk_count=existing["chunk_count"],
                warnings=list(existing["warnings"]),
            )

        stored_path = self.artifacts.store_raw(document_id, content_hash, source)
        document = parser.parse(stored_path, source_path=source)
        document.source_path = source
        document.metadata.update(
            {"original_filename": source.name, "source_size_bytes": size}
        )
        document = self.cleaner.clean(document)
        if not document.blocks:
            raise EmptyDocumentError(f"No indexable text found in: {source}")
        chunks = self.chunker.chunk(document_id, document)
        if not chunks:
            raise EmptyDocumentError(f"No chunks generated from: {source}")
        vectors = self._embed_batches(chunk.text for chunk in chunks)
        if any(not any(vector.dense) or not vector.lexical_weights for vector in vectors):
            raise EmptyDocumentError(f"At least one chunk has no embeddable tokens: {source}")

        processed_path = self.artifacts.store_processed(
            document_id, document, chunks, content_hash, signature
        )
        self.index.replace_document(
            document_id=document_id,
            source_path=str(source),
            stored_path=str(stored_path.resolve()),
            processed_path=str(processed_path.resolve()),
            media_type=document.media_type,
            content_hash=content_hash,
            pipeline_signature=signature,
            embedding_model=self.embedder.name,
            embedding_dimensions=self.embedder.dimensions,
            metadata=document.metadata,
            warnings=document.warnings,
            block_count=len(document.blocks),
            chunks=chunks,
            vectors=vectors,
        )
        return IngestionResult(
            document_id=document_id,
            source_path=str(source),
            stored_path=str(stored_path.resolve()),
            content_hash=content_hash,
            status="indexed",
            block_count=len(document.blocks),
            chunk_count=len(chunks),
            warnings=document.warnings,
        )

    def discover(self, path: str | Path, recursive: bool = True) -> list[Path]:
        target = Path(path).expanduser().resolve()
        if target.is_file():
            return [target]
        if not target.is_dir():
            raise IngestionError(f"Input path does not exist: {target}")
        iterator = target.rglob("*") if recursive else target.glob("*")
        supported = set(self.registry.supported_extensions)
        return sorted(
            (item for item in iterator if item.is_file() and item.suffix.lower() in supported),
            key=lambda item: str(item).casefold(),
        )

    def document_index_status(self, document: dict[str, object]) -> str:
        """Check indexed provenance without mutating the source or Qdrant."""
        if document.get("pipeline_signature") != self.pipeline_signature:
            return "outdated_pipeline"
        source_value = document.get("source_path")
        if not isinstance(source_value, str) or not source_value:
            return "source_missing"
        source = Path(source_value)
        if source.is_symlink() or not source.is_file():
            return "source_missing"
        try:
            if sha256_file(source) != document.get("content_hash"):
                return "source_changed"
        except OSError:
            return "source_missing"
        return "current"

    def _require_current_index(self, document_id: str | None) -> None:
        if document_id is not None:
            document = self.index.get_document(document_id)
            if document is None:
                raise ValueError(f"Document {document_id} is not indexed")
            documents = [document]
        else:
            documents = self.index.documents()
        outdated = [
            (str(document["document_id"]), self.document_index_status(document))
            for document in documents
        ]
        outdated = [(identifier, status) for identifier, status in outdated if status != "current"]
        if outdated:
            details = ", ".join(f"{identifier} ({status})" for identifier, status in outdated[:5])
            if len(outdated) > 5:
                details += f", and {len(outdated) - 5} more"
            raise StaleIndexError(
                "Chỉ mục tài liệu đã cũ hoặc nguồn đã thay đổi; không thể truy vấn: " + details
                + ". Hãy bấm 'Làm mới' trong giao diện hoặc ingest lại file gốc."
            )

    def search(self, query: str, limit: int = 5, document_id: str | None = None) -> list[SearchResult]:
        """Return raw Qdrant hybrid/RRF hits; use retrieve_for_evaluation for reranked hits."""
        if not query.strip():
            return []
        self._require_current_index(document_id)
        vector = self.embedder.embed([query])[0]
        return self.index.search(
            vector,
            limit=limit,
            document_id=document_id,
            embedding_model=self.embedder.name,
            pipeline_signature=self.pipeline_signature,
        )

    def retrieve_for_evaluation(
        self,
        query: str,
        *,
        candidate_limit: int = 30,
        evaluation_limit: int = 10,
        document_id: str | None = None,
    ) -> list[RerankedHit]:
        """Give the future CRAG evaluator cross-encoder-ranked candidate chunks."""
        if self._reranker is None:
            self._reranker = BGEReranker(self.config.reranker_model)
        return RerankedRetriever(self.search, self._reranker).retrieve(
            query,
            candidate_limit=candidate_limit,
            evaluation_limit=evaluation_limit,
            document_id=document_id,
        )

    def select_diverse_context(
        self,
        query: str,
        strips: list[KnowledgeStrip],
        *,
        limit: int,
        min_per_source: dict[str, int] | None = None,
        relevance_weight: float = 0.6,
        duplicate_threshold: float = 0.85,
    ) -> list[KnowledgeStrip]:
        """Select nonredundant refined strips after Correct/Incorrect/Ambiguous routing."""
        return SemanticDiversityFilter(
            self.embedder,
            relevance_weight=relevance_weight,
            duplicate_threshold=duplicate_threshold,
        ).select(query, strips, limit=limit, min_per_source=min_per_source)

    def evaluate_retrieval(
        self,
        query: str,
        *,
        candidate_limit: int = 30,
        evaluation_limit: int = 10,
        document_id: str | None = None,
    ) -> EvaluationDecision:
        """Retrieve, rerank, and judge candidates before CRAG branch execution."""
        hits = self.retrieve_for_evaluation(
            query,
            candidate_limit=candidate_limit,
            evaluation_limit=evaluation_limit,
            document_id=document_id,
        )
        if self._evaluator is None:
            self._evaluator = GeminiRetrievalEvaluator(self.config.evaluator_model)
        return self._evaluator.evaluate(query, hits)

    def refine_internal_knowledge(
        self, query: str, decision: EvaluationDecision
    ) -> list[KnowledgeStrip]:
        """Refine Correct or Ambiguous internal evidence; Incorrect yields no internal strips."""
        if self._evaluator is None:
            self._evaluator = GeminiRetrievalEvaluator(self.config.evaluator_model)
        return KnowledgeRefiner(self._evaluator, self.config.refinement).refine_internal(
            query, decision
        )

    def search_web_knowledge(
        self,
        query: str,
        decision: EvaluationDecision,
        *,
        rewriter: QueryRewriter | None = None,
        provider: WebSearchProvider | None = None,
        fetcher: PageFetcher | None = None,
    ) -> WebKnowledgeResult:
        """Search and verify web evidence only for Incorrect or Ambiguous decisions."""
        if not query.strip():
            raise ValueError("A nonempty query is required for web knowledge search")
        if decision.action == CorrectiveAction.CORRECT:
            return WebKnowledgeResult(decision.action, (), (), (), ())
        if self._reranker is None:
            self._reranker = BGEReranker(self.config.reranker_model)
        if self._evaluator is None:
            self._evaluator = GeminiRetrievalEvaluator(self.config.evaluator_model)
        searcher = WebKnowledgeSearcher(
            rewriter or GeminiQueryRewriter(self.config.evaluator_model),
            provider or DDGSWebSearchProvider(
                region=self.config.web_search.region,
                timeout=self.config.web_search.fetch_timeout,
                attempts=self.config.web_search.search_attempts,
                backoff_seconds=self.config.web_search.search_backoff_seconds,
            ),
            fetcher or SafePageFetcher(self.config.web_search),
            self._reranker,
            self._evaluator,
            config=self.config.web_search,
            refinement=self.config.refinement,
        )
        return searcher.search(query, decision)

    def run_crag(
        self,
        question: str,
        *,
        candidate_limit: int = 30,
        evaluation_limit: int = 10,
        document_id: str | None = None,
    ) -> CragRunResult:
        """Run all three CRAG branches through LangGraph and persist a SQLite checkpoint."""
        from .workflow import CragWorkflow

        self._require_current_index(document_id)
        return CragWorkflow(self).run(
            question, candidate_limit=candidate_limit,
            evaluation_limit=evaluation_limit, document_id=document_id,
        )

    def generate_answer(self, question: str, context: AssembledContext) -> GeneratedAnswer:
        """Generate a citation-checked answer from the assembled evidence."""
        if self._answer_generator is None:
            self._answer_generator = GeminiAnswerGenerator(
                self.config.evaluator_model, config=self.config.answer
            )
        return self._answer_generator.generate(question, context)

    def validate(self, check_files: bool = True) -> ValidationReport:
        return validate_index(self.index, check_files=check_files)

    def _embed_batches(self, texts: Iterable[str], batch_size: int = 32) -> list[EmbeddingVector]:
        result: list[EmbeddingVector] = []
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= batch_size:
                result.extend(self.embedder.embed(batch))
                batch.clear()
        if batch:
            result.extend(self.embedder.embed(batch))
        return result
