from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.colon_fence import colon_fence_plugin
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

from ..models import ParsedDocument, TextBlock
from .base import DocumentParser
from .decoding import read_text_safely

_TASK_ITEM = re.compile(r"^\s*(?:>\s*)*(?:[-+*]|\d+[.)])\s+\[([ xX])\]\s+")
_REFERENCE_USE = re.compile(r"(?<!!)\[([^\]]+)]\[([^\]]+)]")
_REFERENCE_DEF = re.compile(r"^\s{0,3}\[([^\]^]+)]:\s*(\S+)", re.MULTILINE)
_FENCE_OPEN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_TABLE_CELL_DELIMITER = re.compile(r"^:?-+:?$")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.tags: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif tag in {"br", "p", "div", "li", "tr", "blockquote"} and self.parts:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in {"p", "div", "li", "tr", "blockquote"} and self.parts:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return "\n".join(line.strip() for line in "".join(self.parts).splitlines() if line.strip())


class MarkdownParser(DocumentParser):
    """CommonMark parser with GFM and selected widely used extensions."""

    extensions = (".md", ".markdown")

    def __init__(self, image_ocr: Callable[[Path], str] | None = None) -> None:
        self.image_ocr = image_ocr
        self.markdown = (
            MarkdownIt("commonmark", {"html": True})
            .enable(["table", "strikethrough"])
            .use(front_matter_plugin)
            .use(tasklists_plugin, enabled=True)
            .use(footnote_plugin)
            .use(deflist_plugin)
            .use(colon_fence_plugin)
        )

    def parse(self, path: Path, *, source_path: Path | None = None) -> ParsedDocument:
        decoded = read_text_safely(path)
        logical_source = (source_path or path).resolve()
        # markdown-it handles CRLF/CR internally. Keep the decoded source intact
        # so character offsets continue to address the original file exactly.
        source_text = decoded.text
        lines = source_text.splitlines(keepends=True)
        plain_lines = source_text.splitlines()
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        if not lines or offsets[-1] < len(source_text):
            offsets.append(len(source_text))

        environment: dict[str, Any] = {}
        tokens = self.markdown.parse(source_text, environment)
        warnings = list(decoded.warnings)
        metadata: dict[str, Any] = {
            "encoding": decoded.encoding,
            "markdown_dialect": "CommonMark+GFM",
            "extensions": [
                "tables",
                "strikethrough",
                "tasklists",
                "footnotes",
                "definition_lists",
                "colon_fences",
                "front_matter",
            ],
            "reference_definitions": self._reference_definitions(environment),
        }
        blocks = self._tokens_to_blocks(
            tokens,
            source_text,
            plain_lines,
            offsets,
            logical_source,
            metadata,
            warnings,
        )
        self._syntax_warnings(source_text, plain_lines, tokens, warnings)
        return ParsedDocument(
            logical_source,
            "text/markdown",
            blocks,
            metadata=metadata,
            warnings=list(dict.fromkeys(warnings)),
        )

    def _tokens_to_blocks(
        self,
        tokens: list[Token],
        source_text: str,
        lines: list[str],
        offsets: list[int],
        source_path: Path,
        document_metadata: dict[str, Any],
        warnings: list[str],
    ) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        headings: list[str] = []
        list_stack: list[dict[str, Any]] = []
        item_stack: list[dict[str, Any]] = []
        quote_depth = 0
        footnote_label: str | None = None
        definition_kind: str | None = None
        index = 0

        def add(block: TextBlock | None) -> None:
            if block is not None and block.text.strip():
                block.ordinal = len(blocks)
                blocks.append(block)

        while index < len(tokens):
            token = tokens[index]
            token_type = token.type
            if token_type == "front_matter":
                try:
                    value = yaml.safe_load(token.content) or {}
                    document_metadata["front_matter"] = self._json_safe(value)
                    if not isinstance(value, dict):
                        warnings.append("YAML front matter is valid but is not a mapping")
                except yaml.YAMLError as exc:
                    document_metadata["front_matter_raw"] = token.content
                    warnings.append(f"Invalid YAML front matter at line 1: {exc}")
            elif token_type == "heading_open":
                inline = self._next_inline(tokens, index)
                if inline is not None:
                    text, inline_metadata, images = self._render_inline(
                        inline.children or [], source_path, token.map, offsets, warnings
                    )
                    level = int(token.tag[1])
                    headings[:] = headings[: level - 1]
                    headings.append(text.strip())
                    metadata = {
                        "heading_level": level,
                        "heading_style": "setext" if token.markup in {"=", "-"} else "atx",
                        **self._source_span(token.map, offsets, source_path),
                        **inline_metadata,
                    }
                    add(TextBlock(text.strip(), "heading", None, tuple(headings), metadata=metadata))
                    self._add_image_blocks(blocks, images, tuple(headings), metadata)
            elif token_type in {"fence", "code_block"}:
                metadata = {
                    "language": token.info.strip().split(maxsplit=1)[0] if token.info.strip() else None,
                    "info": token.info.strip(),
                    "fence": token.markup or None,
                    "indented": token_type == "code_block",
                    "blockquote_level": quote_depth,
                    **self._source_span(token.map, offsets, source_path),
                }
                add(TextBlock(token.content.rstrip("\n"), "code", None, tuple(headings), metadata=metadata))
            elif token_type == "colon_fence":
                info = token.info.strip()
                match = re.match(r"\{?([\w-]+)}?\s*(.*)", info)
                inline_tokens = self.markdown.parseInline(token.content)
                inline = next((value for value in inline_tokens if value.type == "inline"), None)
                text, inline_metadata, images = self._render_inline(
                    inline.children or [] if inline else [], source_path, token.map, offsets, warnings
                )
                metadata = {
                    "admonition_type": match.group(1) if match else "",
                    "title": match.group(2).strip() if match else "",
                    **self._source_span(token.map, offsets, source_path),
                    **inline_metadata,
                }
                add(TextBlock(text.strip(), "admonition", None, tuple(headings), metadata=metadata))
                self._add_image_blocks(blocks, images, tuple(headings), metadata)
            elif token_type == "html_block":
                extractor = _HTMLTextExtractor()
                extractor.feed(token.content)
                metadata = {
                    "html_tags": list(dict.fromkeys(extractor.tags)),
                    "raw_html": token.content,
                    **self._source_span(token.map, offsets, source_path),
                }
                add(TextBlock(extractor.text, "html", None, tuple(headings), metadata=metadata))
            elif token_type == "table_open":
                block, image_blocks, index = self._table_block(
                    tokens, index, headings, offsets, source_path, warnings
                )
                add(block)
                for image_block in image_blocks:
                    add(image_block)
                continue
            elif token_type in {"bullet_list_open", "ordered_list_open"}:
                start = int(token.attrGet("start") or 1)
                list_stack.append(
                    {
                        "type": "ordered" if token_type == "ordered_list_open" else "bullet",
                        "marker": token.markup,
                        "start": start,
                        "count": 0,
                    }
                )
            elif token_type in {"bullet_list_close", "ordered_list_close"}:
                if list_stack:
                    list_stack.pop()
            elif token_type == "list_item_open":
                if list_stack:
                    list_stack[-1]["count"] += 1
                item_stack.append({"map": token.map, "emitted": False})
            elif token_type == "list_item_close":
                if item_stack:
                    item_stack.pop()
            elif token_type == "blockquote_open":
                quote_depth += 1
            elif token_type == "blockquote_close":
                quote_depth = max(0, quote_depth - 1)
            elif token_type == "footnote_open":
                footnote_label = str(token.meta.get("label", token.meta.get("id", "")))
            elif token_type == "footnote_close":
                footnote_label = None
            elif token_type == "dt_open":
                inline = self._next_inline(tokens, index)
                if inline is not None:
                    line_map = token.map
                    if line_map and line_map[0] == line_map[1]:
                        line_map = [line_map[0], line_map[0] + 1]
                    text, inline_metadata, images = self._render_inline(
                        inline.children or [], source_path, line_map, offsets, warnings
                    )
                    metadata = {
                        **self._source_span(line_map, offsets, source_path),
                        **inline_metadata,
                    }
                    add(TextBlock(text.strip(), "definition_term", None, tuple(headings), metadata=metadata))
                    self._add_image_blocks(blocks, images, tuple(headings), metadata)
                definition_kind = "definition_term"
            elif token_type == "dd_open":
                definition_kind = "definition"
            elif token_type in {"dt_close", "dd_close"}:
                definition_kind = None
            elif token_type == "paragraph_open":
                inline = self._next_inline(tokens, index)
                if inline is not None:
                    line_map = token.map or inline.map or (item_stack[-1]["map"] if item_stack else None)
                    text, inline_metadata, images = self._render_inline(
                        inline.children or [], source_path, line_map, offsets, warnings
                    )
                    metadata = {
                        **self._source_span(line_map, offsets, source_path),
                        **inline_metadata,
                    }
                    kind = "paragraph"
                    if footnote_label is not None:
                        kind = "footnote"
                        metadata["footnote_label"] = footnote_label
                        text = f"[Footnote {footnote_label}] {text}"
                    elif item_stack and list_stack:
                        kind = "list_item"
                        current_list = list_stack[-1]
                        metadata["list"] = {
                            "type": current_list["type"],
                            "marker": current_list["marker"],
                            "level": len(list_stack) - 1,
                            "number": (
                                current_list["start"] + current_list["count"] - 1
                                if current_list["type"] == "ordered"
                                else None
                            ),
                        }
                        source_line = lines[line_map[0]] if line_map and line_map[0] < len(lines) else ""
                        task = _TASK_ITEM.match(source_line)
                        if task:
                            metadata["task"] = {"checked": task.group(1).casefold() == "x"}
                        item_stack[-1]["emitted"] = True
                    elif definition_kind:
                        kind = definition_kind
                    elif quote_depth:
                        kind = "blockquote"
                    if quote_depth:
                        metadata["blockquote_level"] = quote_depth
                    add(TextBlock(text.strip(), kind, None, tuple(headings), metadata=metadata))
                    self._add_image_blocks(blocks, images, tuple(headings), metadata)
            index += 1
        return blocks

    def _render_inline(
        self,
        children: list[Token],
        source_path: Path,
        line_map: list[int] | None,
        offsets: list[int],
        warnings: list[str],
    ) -> tuple[str, dict[str, object], list[dict[str, object]]]:
        text = ""
        formatting_stack: dict[str, list[int]] = {"bold": [], "italic": [], "strike": []}
        formatting: list[dict[str, object]] = []
        link_stack: list[dict[str, object]] = []
        links: list[dict[str, object]] = []
        images: list[dict[str, object]] = []
        references: list[dict[str, str]] = []

        for token in children:
            token_type = token.type
            if token_type in {"text", "code_inline"}:
                start = len(text)
                text += token.content
                if token_type == "code_inline":
                    formatting.append(
                        {"type": "inline_code", "start": start, "end": len(text), "text": token.content}
                    )
            elif token_type in {"softbreak", "hardbreak"}:
                text += "\n"
            elif token_type in {"strong_open", "em_open", "s_open"}:
                name = {"strong_open": "bold", "em_open": "italic", "s_open": "strike"}[token_type]
                formatting_stack[name].append(len(text))
            elif token_type in {"strong_close", "em_close", "s_close"}:
                name = {"strong_close": "bold", "em_close": "italic", "s_close": "strike"}[token_type]
                if formatting_stack[name]:
                    start = formatting_stack[name].pop()
                    formatting.append(
                        {"type": name, "start": start, "end": len(text), "text": text[start:]}
                    )
            elif token_type == "link_open":
                link_stack.append(
                    {
                        "start": len(text),
                        "url": token.attrGet("href") or "",
                        "title": token.attrGet("title") or "",
                    }
                )
            elif token_type == "link_close" and link_stack:
                link = link_stack.pop()
                link["end"] = len(text)
                link["anchor"] = text[int(link["start"]) :]
                links.append(link)
            elif token_type == "image":
                alt = token.content or token.attrGet("alt") or ""
                image = self._image_metadata(
                    token.attrGet("src") or "",
                    alt,
                    token.attrGet("title") or "",
                    source_path,
                    line_map,
                    warnings,
                )
                images.append(image)
                text += alt
            elif token_type == "footnote_ref":
                label = str(token.meta.get("label", token.meta.get("id", "")))
                text += f"[Footnote {label}]"
                references.append({"type": "footnote", "label": label})
            elif token_type == "html_inline":
                if "task-list-item-checkbox" in token.content:
                    continue
                extractor = _HTMLTextExtractor()
                extractor.feed(token.content)
                text += extractor.text

        metadata: dict[str, object] = {}
        if formatting:
            metadata["formatting"] = sorted(formatting, key=lambda item: (int(item["start"]), int(item["end"])))
        if links:
            metadata["links"] = links
        if images:
            metadata["images"] = images
        if references:
            metadata["references"] = references
        return text, metadata, images

    def _image_metadata(
        self,
        source: str,
        alt: str,
        title: str,
        source_path: Path,
        line_map: list[int] | None,
        warnings: list[str],
    ) -> dict[str, object]:
        parsed = urlparse(source)
        line = line_map[0] + 1 if line_map else None
        result: dict[str, object] = {"source": source, "alt_text": alt, "title": title}
        if parsed.scheme or source.startswith("//"):
            result.update({"kind": "remote", "exists": None, "resolved_path": None})
            return result
        clean_path = unquote(parsed.path)
        resolved = (source_path.parent / clean_path).resolve()
        exists = resolved.is_file()
        result.update({"kind": "local", "exists": exists, "resolved_path": str(resolved)})
        if not exists:
            where = f" at line {line}" if line else ""
            warnings.append(f"Markdown image target not found{where}: {source}")
        elif self.image_ocr is not None:
            try:
                result["ocr_text"] = self.image_ocr(resolved).strip()
            except Exception as exc:
                where = f" at line {line}" if line else ""
                warnings.append(f"Markdown image OCR failed{where}: {source}: {exc}")
        return result

    def _add_image_blocks(
        self,
        blocks: list[TextBlock],
        images: list[dict[str, object]],
        heading_path: tuple[str, ...],
        source_metadata: dict[str, object],
    ) -> None:
        for image in images:
            label = str(image.get("alt_text") or Path(str(image.get("source", "image"))).name)
            text = f"[Image: {label}]"
            if image.get("ocr_text"):
                text += f"\n{image['ocr_text']}"
            blocks.append(
                TextBlock(
                    text,
                    "image",
                    None,
                    heading_path,
                    len(blocks),
                    {"image": image, "source": source_metadata.get("source")},
                )
            )

    def _table_block(
        self,
        tokens: list[Token],
        start_index: int,
        headings: list[str],
        offsets: list[int],
        source_path: Path,
        warnings: list[str],
    ) -> tuple[TextBlock, list[TextBlock], int]:
        table_token = tokens[start_index]
        rows: list[list[dict[str, object]]] = []
        row: list[dict[str, object]] | None = None
        cell: dict[str, object] | None = None
        header = False
        image_blocks: list[TextBlock] = []
        index = start_index + 1
        while index < len(tokens) and tokens[index].type != "table_close":
            token = tokens[index]
            if token.type == "thead_open":
                header = True
            elif token.type == "thead_close":
                header = False
            elif token.type == "tr_open":
                row = []
            elif token.type == "tr_close" and row is not None:
                rows.append(row)
                row = None
            elif token.type in {"th_open", "td_open"}:
                style = token.attrGet("style") or ""
                align_match = re.search(r"text-align:\s*(left|right|center)", style)
                cell = {
                    "text": "",
                    "header": token.type == "th_open" or header,
                    "align": align_match.group(1) if align_match else None,
                }
            elif token.type in {"th_close", "td_close"} and cell is not None:
                if row is not None:
                    row.append(cell)
                cell = None
            elif token.type == "inline" and cell is not None:
                text, inline_metadata, images = self._render_inline(
                    token.children or [], source_path, table_token.map, offsets, warnings
                )
                cell["text"] = text
                if inline_metadata:
                    cell["inline"] = inline_metadata
                for image in images:
                    label = str(image.get("alt_text") or image.get("source") or "image")
                    text = f"[Image: {label}]"
                    if image.get("ocr_text"):
                        text += f"\n{image['ocr_text']}"
                    image_blocks.append(
                        TextBlock(
                            text,
                            "image",
                            None,
                            tuple(headings),
                            metadata={"image": image, **self._source_span(table_token.map, offsets, source_path)},
                        )
                    )
            index += 1
        text = "\n".join(
            " | ".join(str(cell_value["text"]) for cell_value in row_value)
            for row_value in rows
        )
        metadata = {
            "table": {"rows": rows},
            **self._source_span(table_token.map, offsets, source_path),
        }
        return TextBlock(text, "table", None, tuple(headings), metadata=metadata), image_blocks, index + 1

    @staticmethod
    def _next_inline(tokens: list[Token], index: int) -> Token | None:
        for candidate in tokens[index + 1 : min(index + 4, len(tokens))]:
            if candidate.type == "inline":
                return candidate
            if candidate.nesting == -1:
                break
        return None

    @staticmethod
    def _source_span(
        line_map: list[int] | None, offsets: list[int], source_path: Path
    ) -> dict[str, object]:
        if not line_map:
            return {}
        start, end = line_map
        start = max(0, min(start, len(offsets) - 1))
        end = max(start, min(end, len(offsets) - 1))
        return {
            "source": {
                "path": str(source_path),
                "start_line": start + 1,
                "end_line": max(start + 1, end),
                "start_char": offsets[start],
                "end_char": offsets[end],
            }
        }

    @staticmethod
    def _reference_definitions(environment: dict[str, Any]) -> dict[str, object]:
        references = environment.get("references", {})
        return MarkdownParser._json_safe(references) if isinstance(references, dict) else {}

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    def _syntax_warnings(
        self,
        source_text: str,
        lines: list[str],
        tokens: list[Token],
        warnings: list[str],
    ) -> None:
        for token in tokens:
            if token.type != "fence" or not token.map:
                continue
            start, end = token.map
            opening = _FENCE_OPEN.match(lines[start]) if start < len(lines) else None
            if not opening:
                continue
            marker = opening.group(1)
            closed = any(
                re.match(rf"^\s{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*$", line)
                for line in lines[start + 1 : end]
            )
            if not closed:
                warnings.append(f"Unclosed fenced code block starting at line {start + 1}")

        if lines and lines[0].strip() == "---" and not any(
            line.strip() in {"---", "..."} for line in lines[1:]
        ):
            warnings.append("Unclosed YAML front matter starting at line 1")

        table_ranges = {
            tuple(token.map)
            for token in tokens
            if token.type == "table_open" and token.map
        }
        for index in range(1, len(lines)):
            delimiter = lines[index]
            if "|" not in lines[index - 1] or "-" not in delimiter:
                continue
            cells = delimiter.strip().strip("|").split("|")
            valid = bool(cells) and all(_TABLE_CELL_DELIMITER.match(cell.strip()) for cell in cells)
            covered = any(start <= index < end for start, end in table_ranges)
            looks_like_delimiter = len(cells) >= 2 and all("-" in cell for cell in cells)
            if looks_like_delimiter and not valid and not covered:
                warnings.append(f"Malformed Markdown table delimiter at line {index + 1}")

        definitions = {label.casefold() for label, _ in _REFERENCE_DEF.findall(source_text)}
        for _, label in _REFERENCE_USE.findall(source_text):
            if label.casefold() not in definitions:
                warnings.append(f"Unresolved Markdown reference link: [{label}]")
