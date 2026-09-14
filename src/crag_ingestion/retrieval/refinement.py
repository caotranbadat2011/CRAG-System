from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace

from ..config import RefinementConfig
from ..models import Chunk, SearchResult
from .diversity import KnowledgeStrip
from .evaluator import CorrectiveAction, EvaluationDecision, Relevance, RetrievalEvaluator
from .service import RerankedHit


_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|\n+")


class KnowledgeRefiner:
    """Decompose internal chunks, judge each strip, and retain citation provenance."""

    def __init__(
        self,
        evaluator: RetrievalEvaluator,
        config: RefinementConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or RefinementConfig()

    def refine_internal(self, query: str, decision: EvaluationDecision) -> list[KnowledgeStrip]:
        if not query.strip():
            raise ValueError("A nonempty query is required for knowledge refinement")
        if decision.action == CorrectiveAction.INCORRECT:
            return []
        allowed = (
            {Relevance.RELEVANT}
            if decision.action == CorrectiveAction.CORRECT
            else {Relevance.RELEVANT, Relevance.UNCERTAIN}
        )
        parents = [item for item in decision.hits if item.relevance in allowed]
        if len({item.hit.result.chunk.chunk_id for item in parents}) != len(parents):
            raise ValueError("Refinement parent chunk IDs must be unique")

        candidates: list[tuple[KnowledgeStrip, RerankedHit]] = []
        for parent in parents:
            result = parent.hit.result
            chunk = result.chunk
            for start, end in self._spans(chunk.text):
                strip_id = f"{chunk.chunk_id}:strip:{start}-{end}"
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_ordinal": chunk.ordinal,
                    "chunk_char_start": start,
                    "chunk_char_end": end,
                    "pages": list(chunk.pages),
                    "heading_path": list(chunk.heading_path),
                    "source_path": result.source_path,
                    "stored_path": result.stored_path,
                    "source_metadata": deepcopy(chunk.metadata.get("source_metadata", [])),
                    "has_overlap": bool(chunk.metadata.get("has_overlap", False)),
                    "parent_relevance": parent.relevance.value,
                    "retrieval_score": result.score,
                    "rerank_score": parent.hit.rerank_score,
                }
                strip = KnowledgeStrip(
                    strip_id=strip_id,
                    text=chunk.text[start:end],
                    source_type="internal",
                    source_ref=chunk.chunk_id,
                    metadata=metadata,
                )
                strip_chunk = Chunk(
                    chunk_id=strip_id,
                    document_id=chunk.document_id,
                    ordinal=chunk.ordinal,
                    text=strip.text,
                    char_count=len(strip.text),
                    pages=chunk.pages,
                    heading_path=chunk.heading_path,
                    metadata={"parent_chunk_id": chunk.chunk_id},
                )
                candidate = RerankedHit(
                    SearchResult(result.score, strip_chunk, result.source_path, result.stored_path),
                    parent.hit.rerank_score,
                )
                candidates.append((strip, candidate))

        relevant: list[KnowledgeStrip] = []
        size = self.config.evaluator_batch_size
        for offset in range(0, len(candidates), size):
            batch = candidates[offset : offset + size]
            judgment = self.evaluator.evaluate(query, [item[1] for item in batch])
            labels = {item.hit.result.chunk.chunk_id: item for item in judgment.hits}
            expected = {item[0].strip_id for item in batch}
            if len(labels) != len(batch) or set(labels) != expected:
                raise ValueError("Strip evaluator returned incomplete or duplicate judgments")
            for strip, _ in batch:
                item = labels[strip.strip_id]
                if item.relevance == Relevance.RELEVANT:
                    relevant.append(replace(
                        strip,
                        metadata={**strip.metadata, "strip_relevance": item.relevance.value,
                                  "strip_reason": item.reason},
                    ))
        return relevant

    def _spans(self, text: str) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        cursor = 0
        for boundary in _BOUNDARY.finditer(text):
            units.extend(self._bounded_span(text, cursor, boundary.start()))
            cursor = boundary.end()
        units.extend(self._bounded_span(text, cursor, len(text)))

        spans: list[tuple[int, int]] = []
        group_start = group_end = None
        count = 0
        for start, end in units:
            if group_start is None:
                group_start, group_end, count = start, end, 1
                continue
            assert group_end is not None
            if (
                count >= self.config.strip_max_sentences
                or end - group_start > self.config.strip_max_chars
                or "\n\n" in text[group_end:start]
            ):
                spans.append((group_start, group_end))
                group_start, group_end, count = start, end, 1
            else:
                group_end, count = end, count + 1
        if group_start is not None and group_end is not None:
            spans.append((group_start, group_end))
        return spans

    def split_spans(self, text: str) -> list[tuple[int, int]]:
        """Return exact half-open offsets for sentence-sized knowledge strips."""
        return self._spans(text)

    def _bounded_span(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        start, end = self._trim(text, start, end)
        spans: list[tuple[int, int]] = []
        limit = self.config.strip_max_chars
        while end - start > limit:
            cut = text.rfind(" ", start + limit // 2, start + limit + 1)
            if cut <= start:
                cut = start + limit
            left = self._trim(text, start, cut)
            if left[0] < left[1]:
                spans.append(left)
            start, _ = self._trim(text, cut, end)
        if start < end:
            spans.append((start, end))
        return spans

    @staticmethod
    def _trim(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end
