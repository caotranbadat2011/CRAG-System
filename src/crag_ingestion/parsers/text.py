from __future__ import annotations

import re
from pathlib import Path

from ..models import ParsedDocument, TextBlock
from .base import DocumentParser
from .decoding import read_text_safely


class TextParser(DocumentParser):
    extensions = (".txt",)

    def parse(self, path: Path, *, source_path: Path | None = None) -> ParsedDocument:
        decoded = read_text_safely(path)
        source = source_path or path
        blocks: list[TextBlock] = []
        for match in re.finditer(r"(.*?)(?:\n[ \t]*\n|\Z)", decoded.text, re.DOTALL):
            raw_text = match.group(1)
            if not raw_text.strip():
                continue
            leading = len(raw_text) - len(raw_text.lstrip())
            trailing = len(raw_text.rstrip())
            start = match.start(1) + leading
            end = match.start(1) + trailing
            text = decoded.text[start:end]
            start_line = decoded.text.count("\n", 0, start) + 1
            end_line = decoded.text.count("\n", 0, end) + 1
            blocks.append(
                TextBlock(
                    text=text,
                    ordinal=len(blocks),
                    metadata={
                        "source": {
                            "start_line": start_line,
                            "end_line": end_line,
                            "start_char": start,
                            "end_char": end,
                        }
                    },
                )
            )
        return ParsedDocument(
            source,
            "text/plain",
            blocks,
            metadata={"encoding": decoded.encoding},
            warnings=decoded.warnings,
        )


# Preserve the original public import path.
from .markdown import MarkdownParser  # noqa: E402

__all__ = ["MarkdownParser", "TextParser"]
