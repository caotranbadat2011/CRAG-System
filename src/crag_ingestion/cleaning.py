from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .config import CleaningConfig
from .models import ParsedDocument, TextBlock

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE = re.compile(r"[^\S\r\n]+")
_MANY_BLANK_LINES = re.compile(r"\n\s*\n(?:\s*\n)+")


class DocumentCleaner:
    def __init__(self, config: CleaningConfig) -> None:
        self.config = config

    def clean(self, document: ParsedDocument) -> ParsedDocument:
        repeated_margins = self._find_repeated_margins(document.blocks)
        cleaned: list[TextBlock] = []
        for block in document.blocks:
            text = self.clean_text(block.text)
            if repeated_margins and block.page is not None:
                lines = [line for line in text.splitlines() if line.strip()]
                while lines and self._margin_key(lines[0]) in repeated_margins:
                    lines.pop(0)
                while lines and self._margin_key(lines[-1]) in repeated_margins:
                    lines.pop()
                text = "\n".join(lines).strip()
            if not text:
                continue
            cleaned.append(
                TextBlock(
                    text=text,
                    kind=block.kind,
                    page=block.page,
                    heading_path=tuple(self.clean_text(h) for h in block.heading_path if h.strip()),
                    ordinal=len(cleaned),
                    metadata=block.metadata.copy(),
                )
            )
        document.blocks = cleaned
        return document

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize(self.config.unicode_form, text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00ad", "").replace("\u00a0", " ")
        text = _CONTROL_CHARACTERS.sub("", text)
        text = _HORIZONTAL_SPACE.sub(" ", text)
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = _MANY_BLANK_LINES.sub("\n\n", text)
        return text.strip()

    def _find_repeated_margins(self, blocks: list[TextBlock]) -> set[str]:
        if not self.config.remove_repeated_margins:
            return set()
        page_lines: dict[int, list[str]] = {}
        for block in blocks:
            if block.page is None:
                continue
            page_lines.setdefault(block.page, []).extend(
                line.strip() for line in block.text.splitlines() if line.strip()
            )
        if len(page_lines) < self.config.repeated_margin_min_pages:
            return set()

        candidates: Counter[str] = Counter()
        for lines in page_lines.values():
            if lines:
                candidates.update({self._margin_key(lines[0]), self._margin_key(lines[-1])})
        threshold = max(
            self.config.repeated_margin_min_pages,
            int(len(page_lines) * self.config.repeated_margin_ratio + 0.999),
        )
        return {line for line, count in candidates.items() if line and count >= threshold}

    @staticmethod
    def _margin_key(line: str) -> str:
        normalized = re.sub(r"\d+", "#", line.casefold())
        return re.sub(r"\s+", " ", normalized).strip()
