from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .config import ChunkingConfig
from .models import Chunk, ParsedDocument, TextBlock
from .utils import stable_chunk_id

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|\n+")


@dataclass(slots=True)
class _Piece:
    text: str
    page: int | None
    heading_path: tuple[str, ...]
    kind: str
    metadata: dict[str, object]
    block_ordinal: int
    block_char_start: int


def slice_source_spans(
    spans: list[dict[str, Any]], start: int, end: int, *, offset: int = 0,
) -> list[dict[str, Any]]:
    """Clip exact chunk ranges while retaining offsets into the cleaned block.

    Parser metadata still describes the original block's source range/bbox;
    these offsets do not claim character precision in the original PDF.
    """
    result: list[dict[str, Any]] = []
    for span in spans:
        left = max(start, span["chunk_char_start"])
        right = min(end, span["chunk_char_end"])
        if left >= right:
            continue
        block_start = span["block_char_start"] + left - span["chunk_char_start"]
        result.append({
            **deepcopy(span),
            "chunk_char_start": offset + left - start,
            "chunk_char_end": offset + right - start,
            "block_char_start": block_start,
            "block_char_end": block_start + right - left,
        })
    return result


class StructuralChunker:
    """Chunk on document structure first, then sentence/character boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document_id: str, document: ParsedDocument) -> list[Chunk]:
        pieces: list[_Piece] = []
        for block in document.blocks:
            pieces.extend(self._split_block(block))
        if not pieces:
            return []

        groups: list[list[_Piece]] = []
        current: list[_Piece] = []
        current_length = 0
        for piece in pieces:
            separator = 2 if current else 0
            if current and current_length + separator + len(piece.text) > self.config.max_chars:
                groups.append(current)
                current = []
                current_length = 0
            current.append(piece)
            current_length += (2 if current_length else 0) + len(piece.text)
        if current:
            groups.append(current)

        # Avoid a tiny trailing chunk when it can be safely merged.
        if len(groups) > 1:
            tail_text = self._join(groups[-1])
            prior_text = self._join(groups[-2])
            if len(tail_text) < self.config.min_chars and len(prior_text) + 2 + len(tail_text) <= self.config.max_chars:
                groups[-2].extend(groups.pop())

        chunks: list[Chunk] = []
        previous_text = ""
        previous_spans: list[dict[str, Any]] = []
        for ordinal, group in enumerate(groups):
            base_text = self._join(group)
            base_spans = self._source_spans(group)
            overlap = self._tail(previous_text, self.config.overlap_chars) if ordinal else ""
            available = self.config.max_chars - len(base_text) - (2 if overlap else 0)
            if overlap and available < len(overlap):
                overlap = self._tail(overlap, max(0, available))
            text = f"{overlap}\n\n{base_text}" if overlap else base_text
            source_spans = slice_source_spans(
                previous_spans, len(previous_text) - len(overlap), len(previous_text),
            ) if overlap else []
            for span in source_spans:
                span["is_overlap"] = True
            source_spans.extend(slice_source_spans(
                base_spans, 0, len(base_text), offset=len(overlap) + 2 if overlap else 0,
            ))
            pages = tuple(sorted({span["page"] for span in source_spans if span["page"] is not None}))
            heading_path = next((tuple(span["heading_path"]) for span in source_spans if span["heading_path"]), ())
            source_metadata: list[dict[str, object]] = []
            for span in source_spans:
                metadata = span["source_metadata"]
                if metadata and metadata not in source_metadata:
                    source_metadata.append(deepcopy(metadata))
            chunk_id = stable_chunk_id(document_id, ordinal, text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    ordinal=ordinal,
                    text=text,
                    char_count=len(text),
                    pages=pages,
                    heading_path=heading_path,
                    metadata={
                        "kinds": sorted({span["kind"] for span in source_spans}),
                        "has_overlap": bool(overlap),
                        "source_metadata": source_metadata,
                        "source_spans": source_spans,
                    },
                )
            )
            previous_text = base_text
            previous_spans = base_spans
        return chunks

    def _split_block(self, block: TextBlock) -> list[_Piece]:
        if len(block.text) <= self.config.max_chars:
            piece = self._piece(block, 0, len(block.text))
            return [piece] if piece else []
        pieces: list[_Piece] = []
        cursor = 0
        ranges: list[tuple[int, int]] = []
        for boundary in _SENTENCE_BOUNDARY.finditer(block.text):
            ranges.append((cursor, boundary.start()))
            cursor = boundary.end()
        ranges.append((cursor, len(block.text)))
        for start, end in ranges:
            while start < end:
                while start < end and block.text[start].isspace():
                    start += 1
                if end - start > self.config.max_chars:
                    cut = block.text.rfind(" ", start, start + self.config.max_chars + 1)
                    if cut - start < self.config.max_chars // 2:
                        cut = start + self.config.max_chars
                else:
                    cut = end
                piece = self._piece(block, start, cut)
                if piece:
                    pieces.append(piece)
                start = cut
        return pieces

    @staticmethod
    def _piece(block: TextBlock, start: int, end: int) -> _Piece | None:
        while start < end and block.text[start].isspace():
            start += 1
        while end > start and block.text[end - 1].isspace():
            end -= 1
        if start == end:
            return None
        return _Piece(
            block.text[start:end], block.page, block.heading_path, block.kind,
            block.metadata, block.ordinal, start,
        )

    @staticmethod
    def _source_spans(group: list[_Piece]) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        cursor = 0
        for piece in group:
            spans.append({
                "chunk_char_start": cursor,
                "chunk_char_end": cursor + len(piece.text),
                "block_ordinal": piece.block_ordinal,
                "block_char_start": piece.block_char_start,
                "block_char_end": piece.block_char_start + len(piece.text),
                "offset_basis": "cleaned_block",
                "page": piece.page,
                "heading_path": list(piece.heading_path),
                "kind": piece.kind,
                "source_metadata": deepcopy(piece.metadata),
                "is_overlap": False,
            })
            cursor += len(piece.text) + 2
        return spans

    @staticmethod
    def _join(group: list[_Piece]) -> str:
        return "\n\n".join(piece.text for piece in group).strip()

    @staticmethod
    def _tail(text: str, limit: int) -> str:
        if limit <= 0 or not text:
            return ""
        if len(text) <= limit:
            return text
        start = text.find(" ", len(text) - limit)
        return text[start + 1 :] if start != -1 else text[-limit:]
