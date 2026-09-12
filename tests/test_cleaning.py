from pathlib import Path

from crag_ingestion.cleaning import DocumentCleaner
from crag_ingestion.config import CleaningConfig
from crag_ingestion.models import ParsedDocument, TextBlock


def test_clean_text_normalizes_unicode_controls_and_whitespace() -> None:
    cleaner = DocumentCleaner(CleaningConfig())
    text = "  Xin\u00a0chào\x00   Việt Nam  \r\n\r\n\r\n  dòng hai  "
    assert cleaner.clean_text(text) == "Xin chào Việt Nam\n\ndòng hai"


def test_repeated_pdf_headers_and_numbered_footers_are_removed() -> None:
    blocks = [
        TextBlock(f"CÔNG TY CRAG\nNội dung trang {page}\nTrang {page}", page=page)
        for page in range(1, 5)
    ]
    document = ParsedDocument(Path("sample.pdf"), "application/pdf", blocks)
    cleaned = DocumentCleaner(CleaningConfig()).clean(document)
    assert [block.text for block in cleaned.blocks] == [
        f"Nội dung trang {page}" for page in range(1, 5)
    ]


def test_margin_removal_is_disabled_for_short_documents() -> None:
    blocks = [TextBlock("Same header\nUnique text", page=page) for page in (1, 2)]
    document = ParsedDocument(Path("short.pdf"), "application/pdf", blocks)
    cleaned = DocumentCleaner(CleaningConfig()).clean(document)
    assert all(block.text.startswith("Same header") for block in cleaned.blocks)

