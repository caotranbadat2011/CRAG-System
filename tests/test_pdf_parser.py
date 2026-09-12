from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from crag_ingestion.exceptions import ParseError
from crag_ingestion.config import ChunkingConfig, IngestionConfig
from crag_ingestion.embeddings.base import Embedder
from crag_ingestion.parsers.pdf import PdfParser
from crag_ingestion.parsers.pdf_layout import extract_page_layout
from crag_ingestion.pipeline import IngestionPipeline


def _build_pdf(path: Path, *, rich: bool = True) -> None:
    image_bytes = b"ff000000ff000000ffffffff>"
    image_object = (
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
        b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /ASCIIHexDecode "
        + f"/Length {len(image_bytes)} >>\nstream\n".encode()
        + image_bytes
        + b"\nendstream"
    )
    if rich:
        operations = b"\n".join(
            [
                b"BT /F2 22 Tf 50 750 Td (System Overview) Tj ET",
                b"BT /F1 10 Tf 50 710 Td (- Left item) Tj ET",
                b"BT /F1 10 Tf 330 710 Td (Right column one) Tj ET",
                b"BT /F1 10 Tf 50 690 Td (1. Left step) Tj ET",
                b"BT /F1 10 Tf 330 690 Td (Right column two) Tj ET",
                b"BT /F1 10 Tf 50 650 Td (E = mc^2) Tj ET",
                b"q 100 0 0 100 250 300 cm /Im1 Do Q",
            ]
        )
    else:
        operations = b"q 500 0 0 700 50 45 cm /Im1 Do Q"
    content_object = (
        f"<< /Length {len(operations)} >>\nstream\n".encode()
        + operations
        + b"\nendstream"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> /XObject << /Im1 6 0 R >> >> "
        b"/Contents 7 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        image_object,
        content_object,
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode())
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(content)


def test_pdf_layout_recovers_headings_styles_lists_formula_columns_and_images(tmp_path: Path) -> None:
    path = tmp_path / "rich.pdf"
    _build_pdf(path)
    logical_source = tmp_path / "original" / "report.pdf"
    document = PdfParser().parse(path, source_path=logical_source)
    kinds: dict[str, list] = {}
    for block in document.blocks:
        kinds.setdefault(block.kind, []).append(block)

    heading = kinds["heading"][0]
    assert heading.text == "System Overview"
    assert heading.metadata["heading_level"] == 1
    assert heading.metadata["font_size"] == 22
    assert heading.metadata["bold"] is True
    assert heading.metadata["source"]["path"] == str(logical_source.resolve())
    assert heading.metadata["bbox"][0] == 50

    assert [block.metadata["list"]["type"] for block in kinds["list_item"]] == [
        "bullet",
        "ordered",
    ]
    assert kinds["list_item"][1].metadata["list"]["number"] == 1
    assert kinds["formula"][0].text == "E = mc^2"
    left = next(block for block in document.blocks if block.text == "- Left item")
    right = next(block for block in document.blocks if block.text == "Right column one")
    assert left.metadata["column"] == 0
    assert right.metadata["column"] == 1
    assert document.blocks.index(left) < document.blocks.index(right)

    image = kinds["image"][0]
    extracted = Path(image.metadata["image"]["extracted_path"])
    assert extracted.is_file()
    assert image.metadata["image"]["sha256"]
    assert document.metadata["image_count"] == 1
    assert document.metadata["source_path"] == str(logical_source.resolve())


def test_image_only_page_creates_image_block_without_text_recognition(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _build_pdf(path, rich=False)
    document = PdfParser().parse(path)
    assert any(block.kind == "image" for block in document.blocks)
    assert not any(block.kind == "ocr" for block in document.blocks)
    assert "ocr_pages" not in document.metadata
    assert any("images but no extractable text" in warning for warning in document.warnings)


def test_encrypted_pdf_accepts_password_and_password_provider(tmp_path: Path) -> None:
    plain = tmp_path / "plain.pdf"
    encrypted = tmp_path / "encrypted.pdf"
    _build_pdf(plain)
    reader = PdfReader(plain)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt("correct-password")
    with encrypted.open("wb") as stream:
        writer.write(stream)

    with pytest.raises(ParseError, match="password is required"):
        PdfParser().parse(encrypted)
    document = PdfParser(password="correct-password").parse(encrypted)
    assert document.metadata["encrypted"] is True

    requested: list[Path] = []

    def provide(source: Path) -> str:
        requested.append(source)
        return "correct-password"

    logical_source = tmp_path / "source.pdf"
    provided = PdfParser(password_provider=provide).parse(
        encrypted, source_path=logical_source
    )
    assert requested == [logical_source.resolve()]
    assert provided.source_path == logical_source.resolve()


class _FakeTable:
    bbox = (40.0, 100.0, 300.0, 180.0)

    @staticmethod
    def extract() -> list[list[str]]:
        return [["Tên", "Tuổi"], ["An", "20"]]


class _FakePlumberPage:
    @staticmethod
    def find_tables() -> list[_FakeTable]:
        return [_FakeTable()]

    @staticmethod
    def extract_words(**_: object) -> list[dict[str, object]]:
        return [
            {"text": "Tên", "x0": 50, "x1": 80, "top": 110, "bottom": 122, "fontname": "Helvetica-Bold", "size": 10},
            {"text": "Tuổi", "x0": 180, "x1": 220, "top": 110, "bottom": 122, "fontname": "Helvetica-Bold", "size": 10},
            {"text": "An", "x0": 50, "x1": 70, "top": 145, "bottom": 157, "fontname": "Helvetica", "size": 10},
            {"text": "20", "x0": 180, "x1": 200, "top": 145, "bottom": 157, "fontname": "Helvetica", "size": 10},
            {"text": "Outside", "x0": 50, "x1": 100, "top": 220, "bottom": 232, "fontname": "Helvetica", "size": 10},
        ]


def test_pdfplumber_table_detection_keeps_structured_rows(tmp_path: Path) -> None:
    path = tmp_path / "page.pdf"
    _build_pdf(path)
    page = PdfReader(path).pages[0]
    blocks, metadata = extract_page_layout(
        page, _FakePlumberPage(), 1, path.resolve()
    )
    table = next(block for block in blocks if block.kind == "table")
    assert table.text == "Tên | Tuổi\nAn | 20"
    assert table.metadata["table"]["rows"] == [["Tên", "Tuổi"], ["An", "20"]]
    assert table.metadata["detection_method"] == "pdfplumber"
    assert metadata["table_count"] == 1
    assert any(block.text == "Outside" for block in blocks)


def test_pdf_pipeline_persists_and_validates_image_provenance(
    tmp_path: Path, fake_embedder: Embedder
) -> None:
    source = tmp_path / "report.pdf"
    _build_pdf(source)
    config = IngestionConfig(
        data_dir=tmp_path / "data",
        chunking=ChunkingConfig(max_chars=400, overlap_chars=40, min_chars=20),
    )
    with IngestionPipeline(config, embedder=fake_embedder) as pipeline:
        result = pipeline.ingest_file(source)
        assert result.status == "indexed"
        report = pipeline.validate()
        assert report.valid, report.to_dict()
        assert report.chunk_count >= 1
