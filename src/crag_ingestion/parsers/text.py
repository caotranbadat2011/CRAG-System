from __future__ import annotations

import re
from pathlib import Path

from ..exceptions import ParseError
from ..models import ParsedDocument, TextBlock
from .base import DocumentParser

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1258", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError(f"Unable to decode text file: {path}")


class TextParser(DocumentParser):
    extensions = (".txt",)

    def parse(self, path: Path) -> ParsedDocument:
        text = _read_text(path)
        paragraphs = re.split(r"\n\s*\n", text)
        blocks = [
            TextBlock(text=p, ordinal=i)
            for i, p in enumerate(paragraphs)
            if p.strip()
        ]
        return ParsedDocument(path, "text/plain", blocks)


class MarkdownParser(DocumentParser):
    extensions = (".md", ".markdown")

    def parse(self, path: Path) -> ParsedDocument:
        lines = _read_text(path).splitlines()
        headings: list[str] = []
        blocks: list[TextBlock] = []
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(
                    TextBlock(text=text, heading_path=tuple(headings), ordinal=len(blocks))
                )
            buffer.clear()

        for line in lines:
            match = _MARKDOWN_HEADING.match(line.strip())
            if match:
                flush()
                level = len(match.group(1))
                headings[:] = headings[: level - 1]
                headings.append(match.group(2).strip())
                blocks.append(
                    TextBlock(
                        text=match.group(2).strip(),
                        kind="heading",
                        heading_path=tuple(headings),
                        ordinal=len(blocks),
                    )
                )
            elif not line.strip():
                flush()
            else:
                buffer.append(line)
        flush()
        return ParsedDocument(path, "text/markdown", blocks)

