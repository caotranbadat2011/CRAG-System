from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TextBlock:
    text: str
    kind: str = "paragraph"
    page: int | None = None
    heading_path: tuple[str, ...] = ()
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        return value


@dataclass(slots=True)
class ParsedDocument:
    source_path: Path
    media_type: str
    blocks: list[TextBlock]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    char_count: int
    pages: tuple[int, ...] = ()
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["pages"] = list(self.pages)
        value["heading_path"] = list(self.heading_path)
        return value


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    source_path: str
    stored_path: str
    content_hash: str
    status: str
    block_count: int
    chunk_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    score: float
    chunk: Chunk
    source_path: str
    stored_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "source_path": self.source_path,
            "stored_path": self.stored_path,
            "chunk": self.chunk.to_dict(),
        }
