from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_QDRANT_COLLECTION = "crag_bge_m3"


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
class IngestionConfig:
    data_dir: Path = Path("data")
    max_file_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 2_000
    pdf_password: str | None = field(default=None, repr=False)
    max_docx_uncompressed_bytes: int = 200 * 1024 * 1024
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    qdrant_url: str | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    qdrant_timeout: int = 30
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    def __post_init__(self) -> None:
        if not self.qdrant_collection.strip():
            raise ValueError("qdrant_collection cannot be empty")
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
