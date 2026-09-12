from __future__ import annotations

import re
from dataclasses import dataclass

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
        for ordinal, group in enumerate(groups):
            base_text = self._join(group)
            overlap = self._tail(previous_text, self.config.overlap_chars) if ordinal else ""
            available = self.config.max_chars - len(base_text) - (2 if overlap else 0)
            if overlap and available < len(overlap):
                overlap = self._tail(overlap, max(0, available))
            text = f"{overlap}\n\n{base_text}" if overlap else base_text
            text = text.strip()
            pages = tuple(sorted({piece.page for piece in group if piece.page is not None}))
            heading_path = next((piece.heading_path for piece in group if piece.heading_path), ())
            source_metadata: list[dict[str, object]] = []
            for piece in group:
                if piece.metadata and piece.metadata not in source_metadata:
                    source_metadata.append(piece.metadata)
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
                        "kinds": sorted({piece.kind for piece in group}),
                        "has_overlap": bool(overlap),
                        "source_metadata": source_metadata,
                    },
                )
            )
            previous_text = base_text
        return chunks

    def _split_block(self, block: TextBlock) -> list[_Piece]:
        if len(block.text) <= self.config.max_chars:
            return [_Piece(block.text, block.page, block.heading_path, block.kind, block.metadata)]
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(block.text) if part.strip()]
        pieces: list[_Piece] = []
        for sentence in sentences or [block.text]:
            while len(sentence) > self.config.max_chars:
                cut = sentence.rfind(" ", 0, self.config.max_chars + 1)
                if cut < self.config.max_chars // 2:
                    cut = self.config.max_chars
                pieces.append(
                    _Piece(
                        sentence[:cut].strip(),
                        block.page,
                        block.heading_path,
                        block.kind,
                        block.metadata,
                    )
                )
                sentence = sentence[cut:].strip()
            if sentence:
                pieces.append(
                    _Piece(sentence, block.page, block.heading_path, block.kind, block.metadata)
                )
        return pieces

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
