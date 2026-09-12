from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import ParsedDocument


class DocumentParser(ABC):
    extensions: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}"

    @abstractmethod
    def parse(self, path: Path, *, source_path: Path | None = None) -> ParsedDocument:
        raise NotImplementedError
