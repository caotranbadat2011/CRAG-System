from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..embeddings.base import Embedder


@dataclass(frozen=True, slots=True)
class KnowledgeStrip:
    """A refined internal or external passage with its citation origin intact."""

    strip_id: str
    text: str
    source_type: str  # "internal" or "web"
    source_ref: str  # chunk ID or source URL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.strip_id.strip(), self.text.strip(), self.source_type.strip(), self.source_ref.strip())):
            raise ValueError("Knowledge strips require an ID, text, source type, and citation reference")


class SemanticDiversityFilter:
    """Select relevant but nonredundant knowledge strips with dense-vector MMR."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        relevance_weight: float = 0.6,
        duplicate_threshold: float = 0.85,
    ) -> None:
        if not 0 <= relevance_weight <= 1:
            raise ValueError("relevance_weight must be between 0 and 1")
        if not 0 < duplicate_threshold <= 1:
            raise ValueError("duplicate_threshold must be between 0 and 1")
        self.embedder = embedder
        self.relevance_weight = relevance_weight
        self.duplicate_threshold = duplicate_threshold

    def select(
        self,
        query: str,
        strips: list[KnowledgeStrip],
        *,
        limit: int,
        min_per_source: Mapping[str, int] | None = None,
    ) -> list[KnowledgeStrip]:
        """Apply MMR after CRAG routing; source quotas can preserve both ambiguous branches."""
        if limit < 0:
            raise ValueError("limit cannot be negative")
        if not query.strip():
            raise ValueError("A nonempty query is required for semantic diversity")
        if not strips or limit == 0:
            return []
        if len({strip.strip_id for strip in strips}) != len(strips):
            raise ValueError("Knowledge strip IDs must be unique")
        quotas = dict(min_per_source or {})
        if any(count < 0 for count in quotas.values()) or sum(quotas.values()) > limit:
            raise ValueError("Source quotas must be nonnegative and fit within limit")
        for source_type, count in quotas.items():
            if sum(strip.source_type == source_type for strip in strips) < count:
                raise ValueError(f"Not enough {source_type} strips to meet source quota")

        embeddings = self.embedder.embed([query, *(strip.text for strip in strips)])
        if len(embeddings) != len(strips) + 1:
            raise ValueError("Embedder returned an inconsistent vector count")
        vectors = [self._unit_vector(item.dense) for item in embeddings]
        query_vector, strip_vectors = vectors[0], vectors[1:]
        relevance = [max(0.0, self._dot(query_vector, vector)) for vector in strip_vectors]
        selected: list[int] = []

        def best_candidate(source_type: str | None = None) -> int | None:
            choices: list[tuple[float, float, int]] = []
            for index, strip in enumerate(strips):
                if index in selected or (source_type is not None and strip.source_type != source_type):
                    continue
                redundancy = max(
                    (max(0.0, self._dot(strip_vectors[index], strip_vectors[chosen]))
                     for chosen in selected),
                    default=0.0,
                )
                if selected and redundancy >= self.duplicate_threshold:
                    continue
                mmr = self.relevance_weight * relevance[index] - (1 - self.relevance_weight) * redundancy
                choices.append((mmr, relevance[index], -index))
            return -max(choices)[2] if choices else None

        for source_type, minimum in quotas.items():
            for _ in range(minimum):
                chosen = best_candidate(source_type)
                if chosen is None:
                    raise ValueError(
                        f"Cannot meet {source_type} quota without near-duplicate strips"
                    )
                selected.append(chosen)

        while len(selected) < min(limit, len(strips)):
            chosen = best_candidate()
            if chosen is None:
                break
            selected.append(chosen)
        return [strips[index] for index in selected]

    @staticmethod
    def _unit_vector(vector: list[float]) -> list[float]:
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Diversity filter requires finite dense vectors")
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm or not math.isfinite(norm):
            raise ValueError("Diversity filter requires nonzero dense vectors")
        return [value / norm for value in vector]

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Diversity vectors have inconsistent dimensions")
        return sum(a * b for a, b in zip(left, right, strict=True))
