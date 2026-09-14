from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..models import TextBlock

_LIST_ITEM = re.compile(r"^\s*([-*•▪◦]|\d+[.)]|[A-Za-z][.)])\s+(.+)$")
_FORMULA = re.compile(r"(?:[=≈≠≤≥±×÷∑∫√∞]|\b(?:sin|cos|tan|log|lim)\s*\()")


def extract_page_layout(
    page: Any,
    plumber_page: Any | None,
    page_number: int,
    source_path: Path,
) -> tuple[list[TextBlock], dict[str, object]]:
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    if plumber_page is not None:
        blocks, metadata = _extract_with_pdfplumber(
            plumber_page, page_number, source_path, width, height
        )
        # Some PDFs encode inter-word spacing through text positioning instead
        # of space glyphs. pdfplumber can then fuse whole phrases into tokens
        # even though pypdf's decoded text preserves the spaces.
        if not metadata["table_count"]:
            plain = page.extract_text() or ""
            extracted = " ".join(block.text for block in blocks)
            if (
                len(plain.split()) >= 80
                and len(plain.split()) >= 1.5 * max(1, len(extracted.split()))
                and len(extracted) >= 0.65 * len(plain)
            ):
                fallback = _extract_pypdf_text_blocks(
                    plain, page_number, source_path, width, height
                )
                if fallback:
                    metadata["backend"] = "pypdf_text_fallback"
                    metadata["position_precision"] = "page"
                    return fallback, metadata
        metadata["backend"] = "pdfplumber"
        return blocks, metadata
    blocks, metadata = _extract_with_pypdf(page, page_number, source_path, width, height)
    metadata["backend"] = "pypdf"
    return blocks, metadata


def classify_document_structure(blocks: list[TextBlock]) -> None:
    text_blocks = [
        block
        for block in blocks
        if block.kind in {"paragraph", "list_item", "formula"}
        and isinstance(block.metadata.get("font_size"), (int, float))
    ]
    weighted_sizes: list[float] = []
    for block in text_blocks:
        size = float(block.metadata["font_size"])
        weighted_sizes.extend([size] * max(1, min(20, len(block.text.split()))))
    body_size = statistics.median(weighted_sizes) if weighted_sizes else 0.0
    candidates: list[TextBlock] = []
    for block in text_blocks:
        size = float(block.metadata["font_size"])
        bbox = block.metadata.get("bbox")
        page_height = float(block.metadata.get("page_height", 0) or 0)
        at_extreme_margin = bool(
            isinstance(bbox, list)
            and page_height
            and (float(bbox[1]) < page_height * 0.015 or float(bbox[3]) > page_height * 0.985)
        )
        looks_like_heading = (
            size >= max(body_size + 1.5, body_size * 1.15)
            and len(block.text) <= 180
            and len(block.text.splitlines()) == 1
            and not block.text.rstrip().endswith((".", ";", ","))
            and not at_extreme_margin
        )
        if looks_like_heading:
            candidates.append(block)
    sizes = sorted({round(float(block.metadata["font_size"]), 2) for block in candidates}, reverse=True)
    headings: list[str] = []
    for block in blocks:
        if block in candidates:
            level = min(6, sizes.index(round(float(block.metadata["font_size"]), 2)) + 1)
            headings[:] = headings[: level - 1]
            headings.append(block.text)
            block.kind = "heading"
            block.metadata["heading_level"] = level
        elif block.kind == "heading":
            level = int(block.metadata.get("heading_level", 1))
            headings[:] = headings[: level - 1]
            headings.append(block.text)
        block.heading_path = tuple(headings)


_SECTION_HEADING = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+([A-Z][^.!?]{2,100})$")


def _extract_pypdf_text_blocks(
    content: str, page_number: int, source_path: Path, width: float, height: float
) -> list[TextBlock]:
    """Recover readable text when glyph positioning defeats word grouping.

    This fallback deliberately advertises page-level provenance: pypdf's
    decoded reading order has no trustworthy word coordinates.
    """
    blocks: list[TextBlock] = []
    current = ""
    source = _source(source_path, page_number, (0.0, 0.0, width, height))

    def emit(value: str, kind: str = "paragraph", level: int = 1) -> None:
        if value:
            metadata: dict[str, object] = {
                "source": dict(source),
                "detection_method": "pypdf_text_fallback",
                "position_precision": "page",
            }
            if kind == "heading":
                metadata["heading_level"] = level
            blocks.append(TextBlock(value, kind, page_number, metadata=metadata))

    def flush() -> None:
        nonlocal current
        emit(current)
        current = ""

    for raw in content.splitlines():
        line = " ".join(raw.split())
        if not line:
            flush()
            continue
        numbered = _SECTION_HEADING.fullmatch(line)
        if numbered or line in {"Abstract", "References", "Conclusion", "Introduction"}:
            flush()
            level = (min(6, numbered.group(1).rstrip(".").count(".") + 1)
                     if numbered else 1)
            emit(line, "heading", max(1, level))
            continue
        if _LIST_ITEM.match(line):
            flush()
            kind, extra = _classify_text_line(line, 0.0, width)
            emit(line, kind)
            if extra:
                blocks[-1].metadata.update(extra)
            continue
        if current.endswith("-") and line[0].islower():
            current = current[:-1] + line
        else:
            current = f"{current} {line}".strip()
        if len(current) >= 600 and current.endswith((".", "?", "!")):
            flush()
    flush()
    return blocks


def _extract_with_pdfplumber(
    page: Any,
    page_number: int,
    source_path: Path,
    width: float,
    height: float,
) -> tuple[list[TextBlock], dict[str, object]]:
    try:
        words = page.extract_words(
            extra_attrs=["fontname", "size"],
            keep_blank_chars=False,
            use_text_flow=False,
        )
    except TypeError:
        words = page.extract_words(extra_attrs=["fontname", "size"])

    table_blocks: list[TextBlock] = []
    table_bboxes: list[tuple[float, float, float, float]] = []
    try:
        tables = page.find_tables()
    except Exception:
        tables = []
    for table in tables:
        rows = table.extract() or []
        normalized_rows = [
            [" ".join((cell or "").split()) for cell in row]
            for row in rows
            if any((cell or "").strip() for cell in row)
        ]
        if not normalized_rows:
            continue
        bbox_values = [float(value) for value in table.bbox]
        bbox = (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])
        table_bboxes.append(bbox)
        text = "\n".join(" | ".join(row) for row in normalized_rows)
        table_blocks.append(
            TextBlock(
                text,
                "table",
                page_number,
                metadata={
                    "table": {"rows": normalized_rows},
                    "bbox": list(bbox),
                    "page_width": width,
                    "page_height": height,
                    "detection_method": "pdfplumber",
                    "source": _source(source_path, page_number, bbox),
                },
            )
        )

    outside_words = [
        word
        for word in words
        if not any(_point_in_bbox(_word_center(word), bbox) for bbox in table_bboxes)
    ]
    segments, split_rows = _word_segments(outside_words, width)
    text_blocks = [_segment_block(segment, page_number, source_path, width, height) for segment in segments]
    combined = _order_columns([*text_blocks, *table_blocks], width, split_rows)
    return combined, {
        "width": width,
        "height": height,
        "word_count": len(words),
        "table_count": len(table_blocks),
        "multi_column": bool(split_rows >= 2),
    }


def _word_segments(words: list[dict[str, Any]], page_width: float) -> tuple[list[dict[str, Any]], int]:
    rows: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda value: (float(value["top"]), float(value["x0"]))):
        for row in rows:
            if abs(float(row[0]["top"]) - float(word["top"])) <= 3.0:
                row.append(word)
                break
        else:
            rows.append([word])

    segments: list[dict[str, Any]] = []
    split_rows = 0
    for row_id, row in enumerate(rows):
        row.sort(key=lambda value: float(value["x0"]))
        current: list[dict[str, Any]] = []
        row_segments: list[list[dict[str, Any]]] = []
        for word in row:
            gap = float(word["x0"]) - float(current[-1]["x1"]) if current else 0.0
            if current and gap > max(36.0, page_width * 0.08):
                row_segments.append(current)
                current = []
            current.append(word)
        if current:
            row_segments.append(current)
        if len(row_segments) >= 2:
            split_rows += 1
        for values in row_segments:
            segments.append({"words": values, "row_id": row_id})
    return segments, split_rows


def _segment_block(
    segment: dict[str, Any],
    page_number: int,
    source_path: Path,
    page_width: float,
    page_height: float,
) -> TextBlock:
    words = segment["words"]
    text = " ".join(str(word["text"]) for word in words).strip()
    bbox = [
        min(float(word["x0"]) for word in words),
        min(float(word["top"]) for word in words),
        max(float(word["x1"]) for word in words),
        max(float(word["bottom"]) for word in words),
    ]
    runs = [
        {
            "text": str(word["text"]),
            "font": str(word.get("fontname") or ""),
            "font_size": float(word.get("size") or 0),
            "bold": _is_bold(str(word.get("fontname") or "")),
            "italic": _is_italic(str(word.get("fontname") or "")),
            "bbox": [float(word["x0"]), float(word["top"]), float(word["x1"]), float(word["bottom"])],
        }
        for word in words
    ]
    font_size = statistics.median([float(str(run["font_size"])) for run in runs])
    kind, extra = _classify_text_line(text, bbox[0], page_width)
    return TextBlock(
        text,
        kind,
        page_number,
        metadata={
            "bbox": bbox,
            "page_width": page_width,
            "page_height": page_height,
            "font_size": font_size,
            "bold": any(bool(run["bold"]) for run in runs),
            "italic": any(bool(run["italic"]) for run in runs),
            "runs": runs,
            "row_id": segment["row_id"],
            "source": _source(
                source_path,
                page_number,
                (bbox[0], bbox[1], bbox[2], bbox[3]),
            ),
            **extra,
        },
    )


def _extract_with_pypdf(
    page: Any,
    page_number: int,
    source_path: Path,
    width: float,
    height: float,
) -> tuple[list[TextBlock], dict[str, object]]:
    fragments = _operation_fragments(page, width, height)

    if not fragments:
        text = page.extract_text() or ""
        for index, line in enumerate(text.splitlines()):
            if line.strip():
                fragments.append(
                    {
                        "text": line.strip(),
                        "x0": 0.0,
                        "x1": width,
                        "top": index * 14.0,
                        "bottom": index * 14.0 + 12.0,
                        "fontname": "",
                        "size": 0.0,
                    }
                )
    segments, split_rows = _word_segments(fragments, width)
    blocks = [_segment_block(segment, page_number, source_path, width, height) for segment in segments]
    return _order_columns(blocks, width, split_rows), {
        "width": width,
        "height": height,
        "fragment_count": len(fragments),
        "table_count": 0,
        "multi_column": bool(split_rows >= 2),
    }


def _operation_fragments(page: Any, width: float, height: float) -> list[dict[str, Any]]:
    try:
        contents = page.get_contents()
        operations = contents.operations if contents is not None else []
        resources = page.get("/Resources", {})
        fonts = resources.get("/Font", {}) if resources else {}
        fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
    except Exception:
        return []

    fragments: list[dict[str, Any]] = []
    ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    graphics_stack: list[list[float]] = []
    text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    line_matrix = list(text_matrix)
    font_name = ""
    font_size = 12.0
    leading = 0.0

    def show(value: object) -> None:
        nonlocal text_matrix
        text = str(value)
        if not text.strip():
            return
        x, y = _transform_point(text_matrix[4], text_matrix[5], ctm)
        estimated_width = max(font_size * 0.45 * len(text), 1.0)
        top = max(0.0, height - y - font_size)
        fragments.append(
            {
                "text": text.strip(),
                "x0": max(0.0, x),
                "x1": min(width, max(0.0, x) + estimated_width),
                "top": top,
                "bottom": min(height, top + font_size * 1.2),
                "fontname": font_name,
                "size": font_size,
            }
        )
        text_matrix[4] += estimated_width

    for operands, operator in operations:
        try:
            if operator == b"q":
                graphics_stack.append(list(ctm))
            elif operator == b"Q" and graphics_stack:
                ctm = graphics_stack.pop()
            elif operator == b"cm" and len(operands) >= 6:
                ctm = _multiply_matrix([float(value) for value in operands[:6]], ctm)
            elif operator == b"BT":
                text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                line_matrix = list(text_matrix)
            elif operator == b"Tf" and len(operands) >= 2:
                font_key = str(operands[0])
                font_size = float(operands[1])
                font_object = fonts.get(font_key, {}) if hasattr(fonts, "get") else {}
                font_object = font_object.get_object() if hasattr(font_object, "get_object") else font_object
                font_name = str(font_object.get("/BaseFont", font_key)).lstrip("/")
            elif operator == b"Tm" and len(operands) >= 6:
                text_matrix = [float(value) for value in operands[:6]]
                line_matrix = list(text_matrix)
            elif operator in {b"Td", b"TD"} and len(operands) >= 2:
                tx, ty = float(operands[0]), float(operands[1])
                if operator == b"TD":
                    leading = -ty
                line_matrix[4] += tx * line_matrix[0] + ty * line_matrix[2]
                line_matrix[5] += tx * line_matrix[1] + ty * line_matrix[3]
                text_matrix = list(line_matrix)
            elif operator == b"TL" and operands:
                leading = float(operands[0])
            elif operator == b"T*":
                line_matrix[4] += -leading * line_matrix[2]
                line_matrix[5] += -leading * line_matrix[3]
                text_matrix = list(line_matrix)
            elif operator == b"Tj" and operands:
                show(operands[0])
            elif operator == b"TJ" and operands:
                values = operands[0]
                combined = ""
                for value in values:
                    if isinstance(value, (int, float)):
                        if float(value) < -150 and combined and not combined.endswith(" "):
                            combined += " "
                    else:
                        combined += str(value)
                show(combined)
            elif operator in {b"'", b'"'} and operands:
                line_matrix[4] += -leading * line_matrix[2]
                line_matrix[5] += -leading * line_matrix[3]
                text_matrix = list(line_matrix)
                show(operands[-1])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return fragments


def _transform_point(x: float, y: float, matrix: list[float]) -> tuple[float, float]:
    return (
        matrix[0] * x + matrix[2] * y + matrix[4],
        matrix[1] * x + matrix[3] * y + matrix[5],
    )


def _multiply_matrix(left: list[float], right: list[float]) -> list[float]:
    return [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
        left[4] * right[0] + left[5] * right[2] + right[4],
        left[4] * right[1] + left[5] * right[3] + right[5],
    ]


def _order_columns(blocks: list[TextBlock], page_width: float, split_rows: int) -> list[TextBlock]:
    if split_rows < 2:
        return sorted(blocks, key=lambda block: (float(block.metadata["bbox"][1]), float(block.metadata["bbox"][0])))
    split_pairs: list[tuple[float, float]] = []
    rows: dict[int, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        row_id = int(block.metadata.get("row_id", -1))
        if row_id >= 0:
            rows[row_id].append(block)
    for row in rows.values():
        ordered = sorted(row, key=lambda block: float(block.metadata["bbox"][0]))
        if len(ordered) >= 2:
            split_pairs.append((float(ordered[0].metadata["bbox"][2]), float(ordered[-1].metadata["bbox"][0])))
    if not split_pairs:
        return blocks
    boundary = statistics.median([(left + right) / 2 for left, right in split_pairs])
    column_top = min(
        float(block.metadata["bbox"][1])
        for block in blocks
        if float(block.metadata["bbox"][2]) <= boundary
        or float(block.metadata["bbox"][0]) >= boundary
    )
    for block in blocks:
        bbox = block.metadata["bbox"]
        if float(bbox[2]) <= boundary:
            block.metadata["column"] = 0
            block.metadata["reading_group"] = 1
        elif float(bbox[0]) >= boundary:
            block.metadata["column"] = 1
            block.metadata["reading_group"] = 2
        else:
            block.metadata["column"] = None
            block.metadata["reading_group"] = 0 if float(bbox[1]) <= column_top else 3
        block.metadata["layout"] = "multi_column"
    return sorted(
        blocks,
        key=lambda block: (
            int(block.metadata.get("reading_group", 0)),
            float(block.metadata["bbox"][1]),
            float(block.metadata["bbox"][0]),
        ),
    )


def _classify_text_line(text: str, x0: float, page_width: float) -> tuple[str, dict[str, object]]:
    list_match = _LIST_ITEM.match(text)
    if list_match:
        marker = list_match.group(1)
        ordered = marker[0].isalnum()
        number_match = re.match(r"\d+", marker)
        number = int(number_match.group()) if number_match is not None else None
        return "list_item", {
            "list": {
                "type": "ordered" if ordered else "bullet",
                "marker": marker,
                "number": number,
                "level": max(
                    0,
                    round(
                        (x0 - page_width * 0.07)
                        / max(24.0, page_width * 0.04)
                    ),
                ),
            }
        }
    if _FORMULA.search(text):
        return "formula", {"formula_detection": "text_heuristic"}
    return "paragraph", {}


def _source(
    source_path: Path, page_number: int, bbox: tuple[float, float, float, float]
) -> dict[str, object]:
    return {"path": str(source_path), "page": page_number, "bbox": list(bbox)}


def _word_center(word: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(word["x0"]) + float(word["x1"])) / 2,
        (float(word["top"]) + float(word["bottom"])) / 2,
    )


def _point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _is_bold(font_name: str) -> bool:
    return any(value in font_name.casefold() for value in ("bold", "black", "demi", "semibold"))


def _is_italic(font_name: str) -> bool:
    return any(value in font_name.casefold() for value in ("italic", "oblique"))
