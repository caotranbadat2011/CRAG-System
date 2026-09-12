from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from crag_ingestion.exceptions import ParseError
from crag_ingestion.parsers.docx import DocxParser
from crag_ingestion.parsers.pdf import PdfParser
from crag_ingestion.parsers.text import MarkdownParser, TextParser


def _write_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Tổng quan</w:t></w:r></w:p>
    <w:p><w:r><w:t>Nội dung CRAG tiếng Việt.</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cột A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Cột B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
  </w:body>
</w:document>"""
    core_xml = """<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Tài liệu mẫu</dc:title></cp:coreProperties>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("docProps/core.xml", core_xml)


def _write_rich_docx(path: Path) -> bytes:
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
 <w:body>
  <w:p><w:pPr><w:pStyle w:val="CustomTitle"/></w:pPr><w:r><w:t>Phần tùy chỉnh</w:t></w:r></w:p>
  <w:p>
   <w:r><w:rPr><w:b/></w:rPr><w:t>Hello</w:t></w:r>
   <w:r><w:rPr><w:i/></w:rPr><w:t>World</w:t></w:r>
   <w:r><w:footnoteReference w:id="2"/></w:r>
   <w:r><w:commentReference w:id="5"/></w:r>
  </w:p>
  <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="7"/></w:numPr></w:pPr><w:r><w:t>Mục bullet</w:t></w:r></w:p>
  <w:p><w:r><w:drawing><wp:docPr id="1" name="Picture 1" descr="Sơ đồ CRAG"/><a:blip r:embed="rId9"/></w:drawing></w:r></w:p>
  <w:p><w:r><w:txbxContent><w:p><w:r><w:t>Nội dung hộp văn bản</w:t></w:r></w:p></w:txbxContent></w:r></w:p>
  <w:tbl><w:tr><w:tc>
   <w:p><w:r><w:t>Ô ngoài</w:t></w:r></w:p>
   <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Ô lồng</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
  </w:tc></w:tr></w:tbl>
 </w:body>
</w:document>"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:style w:type="paragraph" w:styleId="CustomTitle"><w:name w:val="Tiêu đề 2 tùy chỉnh"/></w:style>
</w:styles>"""
    numbering_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:abstractNum w:abstractNumId="3"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/></w:lvl></w:abstractNum>
 <w:num w:numId="7"><w:abstractNumId w:val="3"/></w:num>
</w:numbering>"""
    relationships_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>"""
    header_xml = """<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>Header công ty</w:t></w:r></w:p></w:hdr>"""
    footer_xml = """<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>Footer tài liệu</w:t></w:r></w:p></w:ftr>"""
    footnotes_xml = """<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:footnote w:id="2"><w:p><w:r><w:t>Chi tiết chú thích chân trang</w:t></w:r></w:p></w:footnote></w:footnotes>"""
    comments_xml = """<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:comment w:id="5" w:author="Reviewer" w:initials="RV" w:date="2026-09-12T00:00:00Z"><w:p><w:r><w:t>Hãy kiểm tra đoạn này</w:t></w:r></w:p></w:comment></w:comments>"""
    image = b"\x89PNG\r\n\x1a\nfixture-image"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/numbering.xml", numbering_xml)
        archive.writestr("word/_rels/document.xml.rels", relationships_xml)
        archive.writestr("word/header1.xml", header_xml)
        archive.writestr("word/footer1.xml", footer_xml)
        archive.writestr("word/footnotes.xml", footnotes_xml)
        archive.writestr("word/comments.xml", comments_xml)
        archive.writestr("word/media/image1.png", image)
    return image


def _write_pdf(path: Path) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length 52 >>\nstream\nBT /F1 12 Tf 72 720 Td (CRAG retrieval page one) Tj ET\nendstream",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >>",
        b"<< /Length 52 >>\nstream\nBT /F1 12 Tf 72 720 Td (Vector index page two) Tj ET\nendstream",
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


def test_text_parser_decodes_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "vi.txt"
    path.write_bytes("Đây là tiếng Việt.\n\nĐoạn hai.".encode("utf-8-sig"))
    document = TextParser().parse(path)
    assert [block.text for block in document.blocks] == ["Đây là tiếng Việt.", "Đoạn hai."]


def test_markdown_parser_preserves_heading_hierarchy(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Tổng quan\nNội dung.\n\n## Cài đặt\nCác bước.", encoding="utf-8")
    blocks = MarkdownParser().parse(path).blocks
    assert blocks[-1].heading_path == ("Tổng quan", "Cài đặt")
    assert blocks[-1].text == "Các bước."


def test_docx_parser_extracts_heading_paragraph_table_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    _write_docx(path)
    document = DocxParser().parse(path)
    assert [block.kind for block in document.blocks] == ["heading", "paragraph", "table"]
    assert document.blocks[1].heading_path == ("Tổng quan",)
    assert document.blocks[2].text == "Cột A | Cột B"
    assert document.metadata["title"] == "Tài liệu mẫu"


def test_docx_parser_preserves_rich_structure_and_extracts_media(tmp_path: Path) -> None:
    path = tmp_path / "rich.docx"
    expected_image = _write_rich_docx(path)
    document = DocxParser().parse(path)
    by_kind: dict[str, list] = {}
    for block in document.blocks:
        by_kind.setdefault(block.kind, []).append(block)

    heading = by_kind["heading"][0]
    assert heading.text == "Phần tùy chỉnh"
    assert heading.metadata["heading_level"] == 2
    assert heading.metadata["style_name"] == "Tiêu đề 2 tùy chỉnh"

    paragraph = next(block for block in by_kind["paragraph"] if block.text.startswith("Hello"))
    assert paragraph.text == "Hello World[Footnote 2][Comment 5]"
    assert paragraph.metadata["runs"][0]["bold"] is True
    assert paragraph.metadata["runs"][1]["italic"] is True
    assert paragraph.metadata["references"] == [
        {"type": "footnote", "id": "2"},
        {"type": "comment", "id": "5"},
    ]

    list_item = by_kind["list_item"][0]
    assert list_item.metadata["list"] == {
        "num_id": "7",
        "level": 0,
        "format": "bullet",
        "level_text": "•",
        "start": 1,
    }
    assert by_kind["header"][0].text == "Header công ty"
    assert by_kind["footer"][0].text == "Footer tài liệu"
    assert by_kind["text_box"][0].text == "Nội dung hộp văn bản"
    assert by_kind["footnote"][0].text.endswith("Chi tiết chú thích chân trang")
    assert by_kind["comment"][0].metadata["author"] == "Reviewer"

    table = by_kind["table"][0]
    assert "[Nested table]" in table.text
    nested = table.metadata["table"]["rows"][0][0]["tables"][0]
    assert nested["rows"][0][0]["paragraphs"][0]["text"] == "Ô lồng"

    image = by_kind["image"][0]
    assert image.text == "[Image: Sơ đồ CRAG]"
    extracted = Path(image.metadata["image"]["extracted_path"])
    assert extracted.read_bytes() == expected_image
    assert document.metadata["image_count"] == 1
    assert document.metadata["has_comments"] is True
    assert document.metadata["has_footnotes"] is True


def test_malformed_docx_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.docx"
    path.write_bytes(b"not a zip")
    with pytest.raises(ParseError, match="Invalid DOCX"):
        DocxParser().parse(path)


def test_pdf_parser_extracts_text_and_page_provenance(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    _write_pdf(path)
    document = PdfParser().parse(path)
    assert document.metadata["page_count"] == 2
    assert [block.page for block in document.blocks] == [1, 2]
    assert "CRAG retrieval" in document.blocks[0].text
    assert "Vector index" in document.blocks[1].text
