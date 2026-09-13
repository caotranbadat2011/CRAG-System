from __future__ import annotations

import math
from typing import Any, Protocol

from ..config import DEFAULT_RERANKER_MODEL


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class BGEReranker:
    """Cross-encoder scoring for query/passage pairs; scores are not calibrated probabilities."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def name(self) -> str:
        return self.model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if not query.strip() or any(not passage.strip() for passage in passages):
            raise ValueError("Reranking requires a nonempty query and passages")
        self._ensure_model()
        pairs = [[query, passage] for passage in passages]
        raw_scores = self._model.compute_score(pairs, normalize=False)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]
        scores = [float(value) for value in raw_scores]
        if len(scores) != len(passages) or not all(math.isfinite(value) for value in scores):
            raise ValueError("BGE reranker returned invalid or inconsistent scores")
        return scores

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is required for BGE reranking; reinstall project dependencies"
            ) from exc
        self._model = FlagReranker(self.model_name, use_fp16=False)
