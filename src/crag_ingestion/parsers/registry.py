from __future__ import annotations

from pathlib import Path

from ..config import IngestionConfig
from ..exceptions import UnsupportedFormatError
from .base import DocumentParser
from .docx import DocxParser
from .pdf import PdfParser
from .text import MarkdownParser, TextParser


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._parsers: dict[str, DocumentParser] = {}
        for parser in parsers or []:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        for extension in parser.extensions:
            self._parsers[extension.lower()] = parser

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._parsers))

    def parser_for(self, path: Path) -> DocumentParser:
        parser = self._parsers.get(path.suffix.lower())
        if parser is None:
            supported = ", ".join(self.supported_extensions)
            raise UnsupportedFormatError(f"Unsupported file type '{path.suffix}'. Supported: {supported}")
        return parser


def default_registry(config: IngestionConfig) -> ParserRegistry:
    return ParserRegistry(
        [
            TextParser(),
            MarkdownParser(),
            DocxParser(config.max_docx_uncompressed_bytes),
            PdfParser(config.max_pdf_pages),
        ]
    )
