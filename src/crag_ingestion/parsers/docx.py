from __future__ import annotations

import hashlib
import mimetypes
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from ..exceptions import ParseError
from ..models import ParsedDocument, TextBlock
from .base import DocumentParser

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
_PR = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_FALSE_VALUES = {"0", "false", "off", "none"}
_HEADING_NAMES = re.compile(
    r"(?:heading|titre|überschrift|encabezado|titolo|tiêu\s*đề|nagłówek|заголовок)\s*([1-9])?",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _Style:
    name: str = ""
    based_on: str | None = None
    outline_level: int | None = None
    num_id: str | None = None
    list_level: int | None = None


@dataclass(slots=True)
class _Context:
    styles: dict[str, _Style]
    numbering: dict[tuple[str, int], dict[str, str | int]]
    relationships: dict[tuple[str, str], str]
    assets: dict[str, dict[str, object]]
    repair_run_boundaries: bool

    def resolved_style(self, style_id: str) -> _Style:
        chain: list[_Style] = []
        seen: set[str] = set()
        current = style_id
        while current and current not in seen:
            seen.add(current)
            style = self.styles.get(current)
            if style is None:
                break
            chain.append(style)
            current = style.based_on or ""
        result = _Style()
        for style in reversed(chain):
            result = _Style(
                name=style.name or result.name,
                based_on=style.based_on,
                outline_level=(
                    style.outline_level
                    if style.outline_level is not None
                    else result.outline_level
                ),
                num_id=style.num_id if style.num_id is not None else result.num_id,
                list_level=(
                    style.list_level if style.list_level is not None else result.list_level
                ),
            )
        return result


class DocxParser(DocumentParser):
    extensions = (".docx",)

    def __init__(
        self,
        max_uncompressed_bytes: int = 200 * 1024 * 1024,
        *,
        extract_images: bool = True,
        repair_run_boundaries: bool = True,
    ) -> None:
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.extract_images = extract_images
        self.repair_run_boundaries = repair_run_boundaries

    def parse(self, path: Path, *, source_path: Path | None = None) -> ParsedDocument:
        warnings: list[str] = []
        try:
            with zipfile.ZipFile(path) as archive:
                self._validate_archive(archive)
                main_root = self._required_xml(archive, "word/document.xml", path)
                styles = self._styles(archive, warnings)
                numbering = self._numbering(archive, warnings)
                relationships = self._relationships(archive, warnings)
                assets = self._extract_assets(archive, path, warnings)
                context = _Context(
                    styles,
                    numbering,
                    relationships,
                    assets,
                    self.repair_run_boundaries,
                )

                blocks = self._document_blocks(main_root, "word/document.xml", context)
                for name in sorted(
                    item.filename
                    for item in archive.infolist()
                    if re.fullmatch(r"word/header\d+\.xml", item.filename)
                ):
                    root = self._optional_xml(archive, name, warnings)
                    if root is not None:
                        blocks.extend(self._part_blocks(root, name, context, "header"))
                for name in sorted(
                    item.filename
                    for item in archive.infolist()
                    if re.fullmatch(r"word/footer\d+\.xml", item.filename)
                ):
                    root = self._optional_xml(archive, name, warnings)
                    if root is not None:
                        blocks.extend(self._part_blocks(root, name, context, "footer"))

                footnotes = self._optional_xml(archive, "word/footnotes.xml", warnings)
                if footnotes is not None:
                    blocks.extend(self._note_blocks(footnotes, "word/footnotes.xml", context))
                comments = self._optional_xml(archive, "word/comments.xml", warnings)
                if comments is not None:
                    blocks.extend(self._comment_blocks(comments, "word/comments.xml", context))

                core = self._core_properties(archive)
        except ParseError:
            raise
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            raise ParseError(f"Invalid DOCX file: {path}") from exc

        for ordinal, block in enumerate(blocks):
            block.ordinal = ordinal
        core["images"] = list(assets.values())
        core["image_count"] = len(assets)
        core["has_comments"] = any(block.kind == "comment" for block in blocks)
        core["has_footnotes"] = any(block.kind == "footnote" for block in blocks)
        return ParsedDocument(
            path,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            blocks,
            metadata=core,
            warnings=warnings,
        )

    def _validate_archive(self, archive: zipfile.ZipFile) -> None:
        total = sum(item.file_size for item in archive.infolist())
        if total > self.max_uncompressed_bytes:
            raise ParseError(
                f"DOCX expands to {total} bytes; limit is {self.max_uncompressed_bytes}"
            )
        if "word/document.xml" not in archive.namelist():
            raise ParseError("DOCX does not contain word/document.xml")

    @staticmethod
    def _required_xml(archive: zipfile.ZipFile, name: str, path: Path) -> ET.Element:
        try:
            return ET.fromstring(archive.read(name))
        except (KeyError, ET.ParseError) as exc:
            raise ParseError(f"Invalid WordprocessingML in: {path}") from exc

    @staticmethod
    def _optional_xml(
        archive: zipfile.ZipFile, name: str, warnings: list[str]
    ) -> ET.Element | None:
        try:
            return ET.fromstring(archive.read(name))
        except KeyError:
            return None
        except ET.ParseError:
            warnings.append(f"Ignored malformed optional DOCX part: {name}")
            return None

    def _document_blocks(
        self, root: ET.Element, part_name: str, context: _Context
    ) -> list[TextBlock]:
        body = root.find(f"{_W}body")
        if body is None:
            return []
        return self._container_blocks(body, root, part_name, context)

    def _part_blocks(
        self,
        root: ET.Element,
        part_name: str,
        context: _Context,
        kind_override: str,
    ) -> list[TextBlock]:
        return self._container_blocks(root, root, part_name, context, kind_override)

    def _container_blocks(
        self,
        container: ET.Element,
        complete_root: ET.Element,
        part_name: str,
        context: _Context,
        kind_override: str | None = None,
    ) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        headings: list[str] = []
        for element in container:
            if element.tag == f"{_W}p":
                block = self._paragraph_block(element, headings, context, kind_override)
                if block is not None:
                    blocks.append(block)
            elif element.tag == f"{_W}tbl":
                block = self._table_block(element, headings, context, kind_override)
                if block is not None:
                    blocks.append(block)

        # Text boxes are descendants of drawings and are not direct body children.
        for text_box in complete_root.iter(f"{_W}txbxContent"):
            text_box_headings = list(headings)
            for child in text_box:
                if child.tag == f"{_W}p":
                    block = self._paragraph_block(child, text_box_headings, context, "text_box")
                elif child.tag == f"{_W}tbl":
                    block = self._table_block(child, text_box_headings, context, "text_box")
                else:
                    block = None
                if block is not None:
                    block.metadata["part"] = part_name
                    blocks.append(block)

        blocks.extend(self._image_blocks(complete_root, part_name, tuple(headings), context))
        for block in blocks:
            block.metadata.setdefault("part", part_name)
        return blocks

    def _paragraph_block(
        self,
        paragraph: ET.Element,
        headings: list[str],
        context: _Context,
        kind_override: str | None = None,
    ) -> TextBlock | None:
        text, runs, references = self._paragraph_content(paragraph, context)
        if not text.strip():
            return None
        ppr = paragraph.find(f"{_W}pPr")
        style_id = self._attribute(ppr.find(f"{_W}pStyle") if ppr is not None else None)
        style = context.resolved_style(style_id)
        heading_level = self._heading_level(ppr, style_id, style)
        list_info = self._list_info(ppr, style, context)

        kind = "paragraph"
        if heading_level is not None:
            headings[:] = headings[:heading_level]
            headings.append(text.strip())
            kind = "heading"
        elif list_info is not None:
            kind = "list_item"

        metadata: dict[str, object] = {
            "style_id": style_id,
            "style_name": style.name,
            "runs": runs,
        }
        if heading_level is not None:
            metadata["heading_level"] = heading_level + 1
        if list_info is not None:
            metadata["list"] = list_info
        if references:
            metadata["references"] = references
        if kind_override:
            metadata["content_kind"] = kind
            kind = kind_override
        return TextBlock(text.strip(), kind, None, tuple(headings), 0, metadata)

    def _paragraph_content(
        self, paragraph: ET.Element, context: _Context
    ) -> tuple[str, list[dict[str, object]], list[dict[str, str]]]:
        run_elements: list[ET.Element] = []

        def collect(element: ET.Element) -> None:
            if element.tag == f"{_W}txbxContent":
                return
            if element.tag == f"{_W}r":
                run_elements.append(element)
                return
            for child in element:
                collect(child)

        collect(paragraph)
        text = ""
        runs: list[dict[str, object]] = []
        references: list[dict[str, str]] = []
        for run in run_elements:
            segment, run_references = self._run_text(run)
            if not segment:
                continue
            if context.repair_run_boundaries and self._needs_boundary_space(text, segment):
                segment = " " + segment
            start = len(text)
            text += segment
            formatting = self._run_formatting(run)
            runs.append({"text": segment, "start": start, "end": len(text), **formatting})
            references.extend(run_references)
        return text, runs, references

    @staticmethod
    def _run_text(run: ET.Element) -> tuple[str, list[dict[str, str]]]:
        pieces: list[str] = []
        references: list[dict[str, str]] = []
        for node in run.iter():
            if node.tag == f"{_W}t":
                pieces.append(node.text or "")
            elif node.tag == f"{_W}tab":
                pieces.append("\t")
            elif node.tag in {f"{_W}br", f"{_W}cr"}:
                pieces.append("\n")
            elif node.tag == f"{_W}noBreakHyphen":
                pieces.append("‑")
            elif node.tag == f"{_W}softHyphen":
                pieces.append("\u00ad")
            elif node.tag == f"{_W}footnoteReference":
                note_id = node.get(f"{_W}id", "")
                pieces.append(f"[Footnote {note_id}]")
                references.append({"type": "footnote", "id": note_id})
            elif node.tag == f"{_W}commentReference":
                comment_id = node.get(f"{_W}id", "")
                pieces.append(f"[Comment {comment_id}]")
                references.append({"type": "comment", "id": comment_id})
        return "".join(pieces), references

    @staticmethod
    def _run_formatting(run: ET.Element) -> dict[str, object]:
        rpr = run.find(f"{_W}rPr")
        if rpr is None:
            return {"bold": False, "italic": False, "underline": False, "strike": False}

        def enabled(tag: str) -> bool:
            node = rpr.find(f"{_W}{tag}")
            return node is not None and node.get(f"{_W}val", "true").casefold() not in _FALSE_VALUES

        result: dict[str, object] = {
            "bold": enabled("b") or enabled("bCs"),
            "italic": enabled("i") or enabled("iCs"),
            "underline": enabled("u"),
            "strike": enabled("strike") or enabled("dstrike"),
        }
        style = rpr.find(f"{_W}rStyle")
        if style is not None:
            result["character_style"] = style.get(f"{_W}val", "")
        return result

    @staticmethod
    def _needs_boundary_space(left: str, right: str) -> bool:
        if not left or not right or left[-1].isspace() or right[0].isspace():
            return False
        return (left[-1].islower() and right[0].isupper()) or (
            left[-1] in ".,;:!?" and right[0].isalnum()
        )

    def _heading_level(
        self, ppr: ET.Element | None, style_id: str, style: _Style
    ) -> int | None:
        direct = ppr.find(f"{_W}outlineLvl") if ppr is not None else None
        value = self._int_attribute(direct)
        if value is None:
            value = style.outline_level
        if value is not None and 0 <= value <= 8:
            return value
        label = f"{style_id} {style.name}".strip()
        match = _HEADING_NAMES.search(label)
        if match:
            return max(0, int(match.group(1) or 1) - 1)
        return None

    def _list_info(
        self, ppr: ET.Element | None, style: _Style, context: _Context
    ) -> dict[str, str | int] | None:
        num_pr = ppr.find(f"{_W}numPr") if ppr is not None else None
        num_id = self._attribute(num_pr.find(f"{_W}numId") if num_pr is not None else None)
        level = self._int_attribute(num_pr.find(f"{_W}ilvl") if num_pr is not None else None)
        num_id = num_id or style.num_id
        level = level if level is not None else (style.list_level or 0)
        if not num_id or num_id == "0":
            return None
        definition = context.numbering.get((num_id, level), {})
        return {"num_id": num_id, "level": level, **definition}

    def _table_block(
        self,
        table: ET.Element,
        headings: list[str],
        context: _Context,
        kind_override: str | None = None,
    ) -> TextBlock | None:
        structure = self._table_structure(table, context)
        text = self._render_table(structure)
        if not text.strip():
            return None
        metadata: dict[str, object] = {"table": structure}
        kind = "table"
        if kind_override:
            metadata["content_kind"] = kind
            kind = kind_override
        return TextBlock(text, kind, None, tuple(headings), 0, metadata)

    def _table_structure(
        self, table: ET.Element, context: _Context
    ) -> dict[str, object]:
        rows: list[list[dict[str, object]]] = []
        for row in table.findall(f"{_W}tr"):
            cells: list[dict[str, object]] = []
            for cell in row.findall(f"{_W}tc"):
                paragraphs: list[dict[str, object]] = []
                nested_tables: list[dict[str, object]] = []
                for child in cell:
                    if child.tag == f"{_W}p":
                        text, runs, references = self._paragraph_content(child, context)
                        if text.strip():
                            item: dict[str, object] = {"text": text.strip(), "runs": runs}
                            if references:
                                item["references"] = references
                            paragraphs.append(item)
                    elif child.tag == f"{_W}tbl":
                        nested_tables.append(self._table_structure(child, context))
                cells.append({"paragraphs": paragraphs, "tables": nested_tables})
            rows.append(cells)
        return {"rows": rows}

    def _render_table(self, table: dict[str, object], depth: int = 0) -> str:
        rendered_rows: list[str] = []
        for row in table.get("rows", []):  # type: ignore[union-attr]
            rendered_cells: list[str] = []
            for cell in row:
                parts = [item["text"] for item in cell["paragraphs"]]
                for nested in cell["tables"]:
                    nested_text = self._render_table(nested, depth + 1)
                    if nested_text:
                        indent = "  " * (depth + 1)
                        parts.append("[Nested table]\n" + "\n".join(indent + line for line in nested_text.splitlines()))
                rendered_cells.append(" / ".join(parts))
            rendered_rows.append(" | ".join(rendered_cells))
        return "\n".join(row for row in rendered_rows if row.strip(" |"))

    def _image_blocks(
        self,
        root: ET.Element,
        part_name: str,
        heading_path: tuple[str, ...],
        context: _Context,
    ) -> list[TextBlock]:
        blips = list(root.iter(f"{_A}blip"))
        properties = list(root.iter(f"{_WP}docPr"))
        blocks: list[TextBlock] = []
        for index, blip in enumerate(blips):
            relationship_id = blip.get(f"{_R}embed") or blip.get(f"{_R}link")
            if not relationship_id:
                continue
            target = context.relationships.get((part_name, relationship_id))
            asset = context.assets.get(target or "")
            prop = properties[index] if index < len(properties) else None
            alt = ""
            if prop is not None:
                alt = prop.get("descr") or prop.get("title") or prop.get("name") or ""
            label = alt or (posixpath.basename(target) if target else relationship_id)
            metadata: dict[str, object] = {
                "relationship_id": relationship_id,
                "package_path": target,
                "alt_text": alt,
            }
            if asset:
                metadata.update(asset)
            blocks.append(
                TextBlock(
                    f"[Image: {label}]",
                    "image",
                    None,
                    heading_path,
                    0,
                    {"image": metadata, "part": part_name},
                )
            )
        return blocks

    def _note_blocks(
        self, root: ET.Element, part_name: str, context: _Context
    ) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        for note in root.findall(f"{_W}footnote"):
            note_id = note.get(f"{_W}id", "")
            if note_id.startswith("-") or note_id == "0":
                continue
            parts = [self._paragraph_content(p, context)[0].strip() for p in note.iter(f"{_W}p")]
            text = "\n".join(part for part in parts if part)
            if text:
                blocks.append(
                    TextBlock(
                        f"[Footnote {note_id}] {text}",
                        "footnote",
                        metadata={"note_id": note_id, "part": part_name},
                    )
                )
        blocks.extend(self._image_blocks(root, part_name, (), context))
        return blocks

    def _comment_blocks(
        self, root: ET.Element, part_name: str, context: _Context
    ) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        for comment in root.findall(f"{_W}comment"):
            comment_id = comment.get(f"{_W}id", "")
            parts = [self._paragraph_content(p, context)[0].strip() for p in comment.iter(f"{_W}p")]
            text = "\n".join(part for part in parts if part)
            if not text:
                continue
            blocks.append(
                TextBlock(
                    f"[Comment {comment_id}] {text}",
                    "comment",
                    metadata={
                        "comment_id": comment_id,
                        "author": comment.get(f"{_W}author", ""),
                        "initials": comment.get(f"{_W}initials", ""),
                        "date": comment.get(f"{_W}date", ""),
                        "part": part_name,
                    },
                )
            )
        blocks.extend(self._image_blocks(root, part_name, (), context))
        return blocks

    def _extract_assets(
        self, archive: zipfile.ZipFile, path: Path, warnings: list[str]
    ) -> dict[str, dict[str, object]]:
        assets: dict[str, dict[str, object]] = {}
        media_names = [
            item.filename
            for item in archive.infolist()
            if item.filename.startswith("word/media/") and not item.is_dir()
        ]
        if not media_names:
            return assets
        output_dir = path.parent / f"{path.stem}.media"
        if self.extract_images:
            output_dir.mkdir(parents=True, exist_ok=True)
        for name in media_names:
            try:
                content = archive.read(name)
            except (KeyError, OSError):
                warnings.append(f"Unable to extract DOCX media part: {name}")
                continue
            digest = hashlib.sha256(content).hexdigest()
            suffix = Path(name).suffix.lower() or ".bin"
            destination = output_dir / f"{digest}{suffix}"
            if self.extract_images and not destination.exists():
                destination.write_bytes(content)
            assets[name] = {
                "sha256": digest,
                "content_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
                "size_bytes": len(content),
                "extracted_path": str(destination.resolve()) if self.extract_images else None,
            }
        return assets

    def _relationships(
        self, archive: zipfile.ZipFile, warnings: list[str]
    ) -> dict[tuple[str, str], str]:
        result: dict[tuple[str, str], str] = {}
        for item in archive.infolist():
            name = item.filename
            if not re.fullmatch(r"word/(?:.+/)?_rels/[^/]+\.rels", name):
                continue
            root = self._optional_xml(archive, name, warnings)
            if root is None:
                continue
            rel_directory, rel_filename = posixpath.split(name)
            source_directory = rel_directory.rsplit("/_rels", 1)[0]
            source_name = rel_filename.removesuffix(".rels")
            source_part = posixpath.join(source_directory, source_name)
            for relationship in root.findall(f"{_PR}Relationship"):
                if relationship.get("TargetMode", "").casefold() == "external":
                    continue
                rel_id = relationship.get("Id")
                target = relationship.get("Target")
                if rel_id and target:
                    package_path = posixpath.normpath(
                        posixpath.join(source_directory, unquote(target).lstrip("/"))
                    )
                    result[(source_part, rel_id)] = package_path
        return result

    def _styles(self, archive: zipfile.ZipFile, warnings: list[str]) -> dict[str, _Style]:
        root = self._optional_xml(archive, "word/styles.xml", warnings)
        if root is None:
            return {}
        result: dict[str, _Style] = {}
        for node in root.findall(f"{_W}style"):
            if node.get(f"{_W}type", "paragraph") != "paragraph":
                continue
            style_id = node.get(f"{_W}styleId", "")
            if not style_id:
                continue
            name = self._attribute(node.find(f"{_W}name"))
            based_on = self._attribute(node.find(f"{_W}basedOn")) or None
            ppr = node.find(f"{_W}pPr")
            outline = self._int_attribute(ppr.find(f"{_W}outlineLvl") if ppr is not None else None)
            num_pr = ppr.find(f"{_W}numPr") if ppr is not None else None
            num_id = self._attribute(num_pr.find(f"{_W}numId") if num_pr is not None else None) or None
            level = self._int_attribute(num_pr.find(f"{_W}ilvl") if num_pr is not None else None)
            result[style_id] = _Style(name, based_on, outline, num_id, level)
        return result

    def _numbering(
        self, archive: zipfile.ZipFile, warnings: list[str]
    ) -> dict[tuple[str, int], dict[str, str | int]]:
        root = self._optional_xml(archive, "word/numbering.xml", warnings)
        if root is None:
            return {}
        abstract_levels: dict[str, dict[int, dict[str, str | int]]] = {}
        for abstract in root.findall(f"{_W}abstractNum"):
            abstract_id = abstract.get(f"{_W}abstractNumId", "")
            levels: dict[int, dict[str, str | int]] = {}
            for level in abstract.findall(f"{_W}lvl"):
                level_id = int(level.get(f"{_W}ilvl", "0"))
                levels[level_id] = {
                    "format": self._attribute(level.find(f"{_W}numFmt")) or "unknown",
                    "level_text": self._attribute(level.find(f"{_W}lvlText")),
                    "start": self._int_attribute(level.find(f"{_W}start")) or 1,
                }
            abstract_levels[abstract_id] = levels
        result: dict[tuple[str, int], dict[str, str | int]] = {}
        for number in root.findall(f"{_W}num"):
            num_id = number.get(f"{_W}numId", "")
            abstract_id = self._attribute(number.find(f"{_W}abstractNumId"))
            for level, definition in abstract_levels.get(abstract_id, {}).items():
                result[(num_id, level)] = definition
        return result

    @staticmethod
    def _attribute(node: ET.Element | None) -> str:
        return node.get(f"{_W}val", "") if node is not None else ""

    @staticmethod
    def _int_attribute(node: ET.Element | None) -> int | None:
        value = DocxParser._attribute(node)
        try:
            return int(value) if value else None
        except ValueError:
            return None

    @staticmethod
    def _core_properties(archive: zipfile.ZipFile) -> dict[str, object]:
        try:
            root = ET.fromstring(archive.read("docProps/core.xml"))
        except (KeyError, ET.ParseError):
            return {}
        result: dict[str, object] = {}
        for child in root:
            key = child.tag.rsplit("}", 1)[-1]
            if child.text:
                result[key] = child.text
        return result
