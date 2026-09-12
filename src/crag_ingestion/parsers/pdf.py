from __future__ import annotations

import re
from pathlib import Path

from ..exceptions import ParseError
from ..models import ParsedDocument, TextBlock
from .base import DocumentParser


class PdfParser(DocumentParser):
    extensions = (".pdf",)

    def __init__(self, max_pages: int = 2_000) -> None:
        self.max_pages = max_pages

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ParseError("PDF support requires the 'pypdf' package") from exc

        try:
            reader = PdfReader(str(path), strict=False)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ParseError(f"Encrypted PDF cannot be opened: {path}") from exc
            if len(reader.pages) > self.max_pages:
                raise ParseError(f"PDF has {len(reader.pages)} pages; limit is {self.max_pages}")

            blocks: list[TextBlock] = []
            warnings: list[str] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    warnings.append(f"Page {page_number} contains no extractable text")
                    continue
                for paragraph in re.split(r"\n\s*\n", text):
                    if paragraph.strip():
                        blocks.append(
                            TextBlock(paragraph, "paragraph", page_number, (), len(blocks))
                        )
            metadata = {
                str(key).lstrip("/"): str(value)
                for key, value in (reader.metadata or {}).items()
                if value is not None
            }
            metadata["page_count"] = len(reader.pages)
            return ParsedDocument(path, "application/pdf", blocks, metadata, warnings)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Unable to parse PDF: {path}: {exc}") from exc

