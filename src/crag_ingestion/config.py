from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_EVALUATOR_MODEL = "gemini-3.5-flash-lite"
DEFAULT_QDRANT_COLLECTION = "crag_bge_m3_hybrid"


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    unicode_form: str = "NFC"
    remove_repeated_margins: bool = True
    repeated_margin_min_pages: int = 3
    repeated_margin_ratio: float = 0.6


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_chars: int = 1_200
    overlap_chars: int = 180
    min_chars: int = 120

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("max_chars must be at least 100")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("overlap_chars must be >= 0 and < max_chars")
        if not 0 <= self.min_chars <= self.max_chars:
            raise ValueError("min_chars must be between 0 and max_chars")


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    strip_max_chars: int = 360
    strip_max_sentences: int = 2
    evaluator_batch_size: int = 12

    def __post_init__(self) -> None:
        if self.strip_max_chars < 80:
            raise ValueError("strip_max_chars must be at least 80")
        if self.strip_max_sentences <= 0 or self.evaluator_batch_size <= 0:
            raise ValueError("strip_max_sentences and evaluator_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    max_queries: int = 2
    results_per_query: int = 5
    max_pages: int = 5
    passages_per_page: int = 4
    max_relevant_strips: int = 8
    max_page_bytes: int = 1_000_000
    max_page_chars: int = 40_000
    max_candidate_passages: int = 60
    fetch_timeout: int = 10
    search_attempts: int = 3
    search_backoff_seconds: float = 0.5
    region: str = "vn-vi"
    allowed_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = (
            self.max_queries, self.results_per_query, self.max_pages,
            self.passages_per_page, self.max_relevant_strips,
            self.max_page_bytes, self.max_page_chars, self.max_candidate_passages,
            self.fetch_timeout,
            self.search_attempts,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("Web search limits and timeout must be positive")
        if self.max_queries > 5 or self.max_pages > 20:
            raise ValueError("Web search query and page limits are too high")
        if self.search_attempts > 5 or not 0 <= self.search_backoff_seconds <= 5:
            raise ValueError("Web search retry settings are out of range")
        if not self.region.strip():
            raise ValueError("Web search region cannot be empty")
        if any(not domain.strip() for domain in self.allowed_domains):
            raise ValueError("Allowed web domains cannot be empty")


@dataclass(frozen=True, slots=True)
class ContextConfig:
    max_strips: int = 8
    max_chars: int = 6_000
    relevance_weight: float = 0.6
    duplicate_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.max_strips <= 0 or self.max_chars <= 0:
            raise ValueError("Context strip and character limits must be positive")
        if not 0 <= self.relevance_weight <= 1:
            raise ValueError("Context relevance_weight must be between 0 and 1")
        if not 0 < self.duplicate_threshold <= 1:
            raise ValueError("Context duplicate_threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AnswerConfig:
    max_claims: int = 12
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        if not 1 <= self.max_claims <= 20:
            raise ValueError("Answer max_claims must be between 1 and 20")
        if not 512 <= self.max_output_tokens <= 8192:
            raise ValueError("Answer max_output_tokens must be between 512 and 8192")


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    data_dir: Path = Path("data")
    max_file_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 2_000
    pdf_password: str | None = field(default=None, repr=False)
    max_docx_uncompressed_bytes: int = 200 * 1024 * 1024
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL
    qdrant_url: str | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    qdrant_timeout: int = 30
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)

    def __post_init__(self) -> None:
        if not self.qdrant_collection.strip():
            raise ValueError("qdrant_collection cannot be empty")
        if not self.evaluator_model.strip():
            raise ValueError("evaluator_model cannot be empty")
        if self.qdrant_timeout <= 0:
            raise ValueError("qdrant_timeout must be positive")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints" / "crag.sqlite3"
