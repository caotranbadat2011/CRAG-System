from __future__ import annotations

from pathlib import Path

from crag_ingestion.parsers.markdown import MarkdownParser
from crag_ingestion.parsers.text import TextParser


def _by_kind(document: object) -> dict[str, list]:
    result: dict[str, list] = {}
    for block in document.blocks:  # type: ignore[attr-defined]
        result.setdefault(block.kind, []).append(block)
    return result


def test_commonmark_headings_code_fences_and_indented_code(tmp_path: Path) -> None:
    path = tmp_path / "syntax.md"
    path.write_text(
        "Tiêu đề Setext\n"
        "===============\n\n"
        "    # Đây là code thụt lề\n\n"
        "```python\n"
        "# Đây là comment Python\n"
        "print('ok')\n"
        "```\n",
        encoding="utf-8",
    )
    document = MarkdownParser().parse(path)
    kinds = _by_kind(document)
    assert [block.text for block in kinds["heading"]] == ["Tiêu đề Setext"]
    assert kinds["heading"][0].metadata["heading_style"] == "setext"
    assert len(kinds["code"]) == 2
    assert kinds["code"][0].metadata["indented"] is True
    assert kinds["code"][1].metadata["language"] == "python"
    assert "# Đây là comment Python" in kinds["code"][1].text
    assert not any("comment Python" in block.text for block in kinds["heading"])


def test_lists_nested_lists_and_task_status_are_structured(tmp_path: Path) -> None:
    path = tmp_path / "lists.md"
    path.write_text(
        "- Mục một\n"
        "- [x] Đã hoàn thành\n"
        "- [ ] Chưa hoàn thành\n"
        "- Mục cha\n"
        "  - Mục lồng một\n"
        "  - Mục lồng hai\n"
        "3. Bước ba\n"
        "4. Bước bốn\n",
        encoding="utf-8",
    )
    document = MarkdownParser().parse(path)
    items = _by_kind(document)["list_item"]
    assert [item.text for item in items[:3]] == ["Mục một", "Đã hoàn thành", "Chưa hoàn thành"]
    assert items[0].metadata["list"]["type"] == "bullet"
    assert items[1].metadata["task"] == {"checked": True}
    assert items[2].metadata["task"] == {"checked": False}
    nested = [item for item in items if item.metadata["list"]["level"] == 1]
    assert [item.text for item in nested] == ["Mục lồng một", "Mục lồng hai"]
    assert all(item.metadata["list"]["type"] == "bullet" for item in nested)
    ordered = [item for item in items if item.metadata["list"]["type"] == "ordered"]
    assert [item.metadata["list"]["number"] for item in ordered] == [3, 4]


def test_table_blockquote_definition_and_admonition_extensions(tmp_path: Path) -> None:
    path = tmp_path / "structures.md"
    path.write_text(
        "| Tên | Tuổi |\n"
        "|:---|---:|\n"
        "| An | 20 |\n\n"
        "> Cấp một\n"
        "> > Cấp hai\n\n"
        "Thuật ngữ\n"
        ": Nội dung định nghĩa\n\n"
        ":::{warning} Cẩn thận\n"
        "**Không xóa dữ liệu.**\n"
        ":::\n",
        encoding="utf-8",
    )
    document = MarkdownParser().parse(path)
    kinds = _by_kind(document)
    table = kinds["table"][0]
    assert table.text == "Tên | Tuổi\nAn | 20"
    assert table.metadata["table"]["rows"][0][0] == {
        "text": "Tên",
        "header": True,
        "align": "left",
    }
    assert table.metadata["table"]["rows"][0][1]["align"] == "right"
    assert [block.metadata["blockquote_level"] for block in kinds["blockquote"]] == [1, 2]
    assert kinds["definition_term"][0].text == "Thuật ngữ"
    assert kinds["definition"][0].text == "Nội dung định nghĩa"
    assert kinds["admonition"][0].metadata["admonition_type"] == "warning"
    assert kinds["admonition"][0].metadata["title"] == "Cẩn thận"
    assert kinds["admonition"][0].text == "Không xóa dữ liệu."


def test_front_matter_inline_formatting_links_images_html_and_footnotes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "diagram.png"
    image_path.write_bytes(b"png fixture")
    path = tmp_path / "rich.md"
    path.write_text(
        "---\n"
        "title: Tài liệu\n"
        "author: Nguyễn Văn A\n"
        "tags: [crag, rag]\n"
        "---\n\n"
        "# Nội dung\n\n"
        "**Đậm**, *nghiêng*, ~~gạch~~ và `inline_code`.\n\n"
        "[OpenAI](https://openai.com) và [Xem tài liệu][docs].\n\n"
        "![Sơ đồ](images/diagram.png \"Kiến trúc\")\n\n"
        "Đây là HTML <strong>nhúng</strong>.\n\n"
        "<div><strong>Nội dung HTML block</strong><script>bad()</script></div>\n\n"
        "Nội dung có chú thích[^1].\n\n"
        "[docs]: https://example.com/docs \"Docs\"\n"
        "[^1]: Chi tiết chú thích\n",
        encoding="utf-8",
    )
    document = MarkdownParser().parse(path)
    kinds = _by_kind(document)
    assert document.metadata["front_matter"] == {
        "title": "Tài liệu",
        "author": "Nguyễn Văn A",
        "tags": ["crag", "rag"],
    }
    assert "title: Tài liệu" not in "\n".join(block.text for block in document.blocks)

    formatted = next(block for block in kinds["paragraph"] if block.text.startswith("Đậm"))
    assert formatted.text == "Đậm, nghiêng, gạch và inline_code."
    assert {item["type"] for item in formatted.metadata["formatting"]} == {
        "bold",
        "italic",
        "strike",
        "inline_code",
    }
    linked = next(block for block in kinds["paragraph"] if block.text.startswith("OpenAI"))
    assert [link["url"] for link in linked.metadata["links"]] == [
        "https://openai.com",
        "https://example.com/docs",
    ]
    assert "DOCS" in document.metadata["reference_definitions"]

    image = kinds["image"][0]
    assert image.text == "[Image: Sơ đồ]"
    assert image.metadata["image"]["alt_text"] == "Sơ đồ"
    assert image.metadata["image"]["exists"] is True
    assert image.metadata["image"]["resolved_path"] == str(image_path.resolve())
    assert kinds["html"][0].text == "Nội dung HTML block"
    assert "bad()" not in kinds["html"][0].text
    assert "<strong>" not in next(block.text for block in kinds["paragraph"] if "HTML" in block.text)
    assert kinds["footnote"][0].metadata["footnote_label"] == "1"
    assert "[Footnote 1]" in next(block.text for block in kinds["paragraph"] if "chú thích" in block.text)


def test_source_provenance_uses_original_path_and_exact_ranges(tmp_path: Path) -> None:
    stored = tmp_path / "stored.md"
    original = tmp_path / "source" / "original.md"
    stored.write_text("# Tiêu đề\n\nĐoạn nội dung.\n", encoding="utf-8")
    document = MarkdownParser().parse(stored, source_path=original)
    heading, paragraph = document.blocks
    source_text = stored.read_bytes().decode("utf-8")
    assert heading.metadata["source"] == {
        "path": str(original.resolve()),
        "start_line": 1,
        "end_line": 1,
        "start_char": 0,
        "end_char": source_text.index("\n") + 1,
    }
    assert paragraph.metadata["source"]["start_line"] == 3
    assert paragraph.metadata["source"]["start_char"] == source_text.index("Đoạn")
    assert paragraph.metadata["source"]["end_char"] == len(source_text)


def test_crlf_offsets_and_optional_image_ocr_are_preserved(tmp_path: Path) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image")
    path = tmp_path / "windows.md"
    raw = b"# Heading\r\n\r\n![Diagram](diagram.png)\r\n"
    path.write_bytes(raw)
    document = MarkdownParser(image_ocr=lambda value: f"OCR:{value.name}").parse(path)
    heading = document.blocks[0]
    image_block = next(block for block in document.blocks if block.kind == "image")
    assert heading.metadata["source"]["end_char"] == len(b"# Heading\r\n".decode())
    assert image_block.metadata["image"]["ocr_text"] == "OCR:diagram.png"
    assert image_block.text == "[Image: Diagram]\nOCR:diagram.png"


def test_warnings_cover_unclosed_fence_bad_table_missing_image_and_reference(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text(
        "| A | B |\n"
        "| -- | broken- |\n\n"
        "![Mất](missing.png)\n\n"
        "[Không có][missing-ref]\n\n"
        "```python\n"
        "print('never closed')\n",
        encoding="utf-8",
    )
    document = MarkdownParser().parse(path)
    warnings = "\n".join(document.warnings)
    assert "Malformed Markdown table" in warnings
    assert "image target not found" in warnings
    assert "Unresolved Markdown reference" in warnings
    assert "Unclosed fenced code block" in warnings


def test_legacy_encoding_is_not_guessed_as_utf16(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes("Café résumé".encode("cp1252"))
    document = TextParser().parse(path)
    assert document.blocks[0].text == "Café résumé"
    assert not str(document.metadata["encoding"]).startswith("utf-16")
    assert document.warnings


def test_invalid_and_unclosed_front_matter_emit_warnings(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid-front-matter.md"
    invalid.write_text("---\ntitle: [broken\n---\n\nBody\n", encoding="utf-8")
    invalid_document = MarkdownParser().parse(invalid)
    assert any("Invalid YAML front matter" in warning for warning in invalid_document.warnings)
    assert "front_matter_raw" in invalid_document.metadata

    unclosed = tmp_path / "unclosed-front-matter.md"
    unclosed.write_text("---\ntitle: Missing close\nBody\n", encoding="utf-8")
    unclosed_document = MarkdownParser().parse(unclosed)
    assert any("Unclosed YAML front matter" in warning for warning in unclosed_document.warnings)
