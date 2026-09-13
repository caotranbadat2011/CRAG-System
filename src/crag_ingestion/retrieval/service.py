from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ..models import SearchResult
from .reranker import Reranker


@dataclass(frozen=True, slots=True)
class RerankedHit:
    result: SearchResult
    rerank_score: float

    @property
    def retrieval_score(self) -> float:
        return self.result.score

    def to_dict(self) -> dict[str, object]:
        return {
            **self.result.to_dict(),
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
        }


class RerankedRetriever:
    """Hybrid candidate retrieval followed by cross-encoder ranking for CRAG evaluation."""

    def __init__(
        self,
        search: Callable[..., list[SearchResult]],
        reranker: Reranker,
    ) -> None:
        self.search = search
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        candidate_limit: int = 30,
        evaluation_limit: int = 10,
        document_id: str | None = None,
    ) -> list[RerankedHit]:
        if candidate_limit <= 0 or evaluation_limit <= 0:
            raise ValueError("Retrieval limits must be positive")
        if evaluation_limit > candidate_limit:
            raise ValueError("evaluation_limit cannot exceed candidate_limit")
        if not query.strip():
            return []
        hits = self.search(query, limit=candidate_limit, document_id=document_id)
        if not hits:
            return []
        scores = self.reranker.score(query, [hit.chunk.text for hit in hits])
        if len(scores) != len(hits) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker must return one finite score per retrieved chunk")
        ranked = sorted(
            (RerankedHit(hit, score) for hit, score in zip(hits, scores, strict=True)),
            key=lambda item: item.rerank_score,
            reverse=True,
        )
        return ranked[:evaluation_limit]
