from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import ContextConfig
from .diversity import KnowledgeStrip
from .evaluator import CorrectiveAction


class ContextSelector(Protocol):
    def __call__(
        self,
        query: str,
        strips: list[KnowledgeStrip],
        *,
        limit: int,
        min_per_source: dict[str, int] | None,
        relevance_weight: float,
        duplicate_threshold: float,
    ) -> list[KnowledgeStrip]: ...


@dataclass(frozen=True, slots=True)
class Citation:
    marker: str
    strip_id: str
    source_type: str
    source_ref: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "strip_id": self.strip_id,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class AssembledContext:
    status: str  # ready, partial, or no_evidence
    text: str
    strips: tuple[KnowledgeStrip, ...]
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "context_text": self.text,
            "selected_strips": [strip.to_dict() for strip in self.strips],
            "citations": [citation.to_dict() for citation in self.citations],
            "warnings": list(self.warnings),
        }


class ContextAssembler:
    """Choose branch-appropriate evidence and make citation-addressable JSONL context."""

    def __init__(self, select: ContextSelector, config: ContextConfig | None = None) -> None:
        self.select = select
        self.config = config or ContextConfig()

    def assemble(
        self,
        question: str,
        action: CorrectiveAction,
        internal: list[KnowledgeStrip],
        web: list[KnowledgeStrip],
    ) -> AssembledContext:
        if not question.strip():
            raise ValueError("Context assembly requires a nonempty question")
        if action == CorrectiveAction.CORRECT and web:
            raise ValueError("Correct branch cannot contain web evidence")
        if action == CorrectiveAction.INCORRECT and internal:
            raise ValueError("Incorrect branch cannot contain internal evidence")
        if any(strip.source_type != "internal" for strip in internal):
            raise ValueError("Internal evidence has the wrong source type")
        if any(strip.source_type != "web" for strip in web):
            raise ValueError("Web evidence has the wrong source type")
        candidates = [*internal, *web]
        if len({strip.strip_id for strip in candidates}) != len(candidates):
            raise ValueError("Evidence strip IDs must be unique across sources")
        if not candidates:
            return AssembledContext("no_evidence", "", (), (), ("No relevant evidence was found",))

        warnings: list[str] = []
        quotas: dict[str, int] | None = None
        if action == CorrectiveAction.AMBIGUOUS:
            if not internal or not web:
                warnings.append("Ambiguous branch lacks evidence from one source")
            elif self.config.max_strips < 2:
                warnings.append("Context strip limit cannot preserve both evidence sources")
            else:
                quotas = {"internal": 1, "web": 1}
        options = {
            "limit": self.config.max_strips,
            "min_per_source": quotas,
            "relevance_weight": self.config.relevance_weight,
            "duplicate_threshold": self.config.duplicate_threshold,
        }
        try:
            selected = self.select(question, candidates, **options)
        except ValueError as exc:
            if quotas is None or "quota" not in str(exc).lower():
                raise
            warnings.append("Both sources could not be preserved without near-duplicate evidence")
            selected = self.select(question, candidates, **{**options, "min_per_source": None})
        eligible = {strip.strip_id: strip for strip in candidates}
        if (
            len(selected) > self.config.max_strips
            or len({strip.strip_id for strip in selected}) != len(selected)
            or any(
                strip.strip_id not in eligible or strip != eligible[strip.strip_id]
                for strip in selected
            )
        ):
            raise ValueError("Context selector returned too much, duplicate, modified, or unknown evidence")

        lines: list[str] = []
        included: list[KnowledgeStrip] = []
        citations: list[Citation] = []
        length = 0
        for strip in selected:
            marker = f"[{len(included) + 1}]"
            line = json.dumps({
                "citation": marker,
                "source_type": strip.source_type,
                "source_ref": strip.source_ref,
                "text": strip.text,
            }, ensure_ascii=False, separators=(",", ":"))
            extra = len(line) + (1 if lines else 0)
            if length + extra > self.config.max_chars:
                warnings.append(f"Skipped {strip.strip_id}: context character budget exceeded")
                continue
            lines.append(line)
            length += extra
            included.append(strip)
            citations.append(Citation(
                marker, strip.strip_id, strip.source_type, strip.source_ref,
                dict(strip.metadata),
            ))

        if not included:
            status = "no_evidence"
            warnings.append("No evidence fits the context character budget")
        elif action == CorrectiveAction.AMBIGUOUS and {
            strip.source_type for strip in included
        } != {"internal", "web"}:
            status = "partial"
            warnings.append("Ambiguous context contains only one evidence source")
        else:
            status = "ready"
        return AssembledContext(
            status, "\n".join(lines), tuple(included), tuple(citations), tuple(warnings)
        )
