from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import ParsedDocument


class DocumentParser(ABC):
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        raise NotImplementedError

