from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .models import Chunk, SearchResult
from .retrieval import (
    CorrectiveAction, EvaluatedHit, EvaluationDecision, GeneratedAnswer,
    KnowledgeStrip, Relevance,
    RerankedHit,
)
from .retrieval.context import AssembledContext, Citation, ContextAssembler

if TYPE_CHECKING:
    from .pipeline import IngestionPipeline


class _GraphState(TypedDict, total=False):
    question: str
    candidate_limit: int
    evaluation_limit: int
    document_id: str | None
    decision: dict[str, Any]
    internal_strips: list[dict[str, Any]]
    web_strips: list[dict[str, Any]]
    web_queries: list[str]
    source_urls: list[str]
    warnings: list[str]
    context: dict[str, Any]
    answer: dict[str, Any]
    answer_pending: bool


def _decision_from_dict(value: dict[str, Any]) -> EvaluationDecision:
    hits: list[EvaluatedHit] = []
    for row in value["hits"]:
        chunk_data = row["chunk"]
        chunk = Chunk(
            chunk_id=chunk_data["chunk_id"],
            document_id=chunk_data["document_id"],
            ordinal=chunk_data["ordinal"],
            text=chunk_data["text"],
            char_count=chunk_data["char_count"],
            pages=tuple(chunk_data.get("pages", ())),
            heading_path=tuple(chunk_data.get("heading_path", ())),
            metadata=chunk_data.get("metadata", {}),
        )
        result = SearchResult(row["score"], chunk, row["source_path"], row["stored_path"])
        hits.append(EvaluatedHit(
            RerankedHit(result, row["rerank_score"]), Relevance(row["relevance"]),
            row["reason"],
        ))
    return EvaluationDecision(CorrectiveAction(value["action"]), value["model"], tuple(hits))


def _strip_from_dict(value: dict[str, Any]) -> KnowledgeStrip:
    return KnowledgeStrip(
        value["strip_id"], value["text"], value["source_type"],
        value["source_ref"], value.get("metadata", {}),
    )


def _context_from_dict(value: dict[str, Any]) -> AssembledContext:
    return AssembledContext(
        value["status"], value["context_text"],
        tuple(_strip_from_dict(item) for item in value["selected_strips"]),
        tuple(Citation(
            item["marker"], item["strip_id"], item["source_type"],
            item["source_ref"], item["metadata"],
        ) for item in value["citations"]),
        tuple(value["warnings"]),
    )


@dataclass(frozen=True, slots=True)
class CragRunResult:
    run_id: str
    checkpoint_path: str
    decision: EvaluationDecision
    internal_strips: tuple[KnowledgeStrip, ...]
    web_strips: tuple[KnowledgeStrip, ...]
    web_queries: tuple[str, ...]
    source_urls: tuple[str, ...]
    context: AssembledContext
    answer: GeneratedAnswer | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint_path": self.checkpoint_path,
            "decision": self.decision.to_dict(),
            "internal_strips": [strip.to_dict() for strip in self.internal_strips],
            "web_strips": [strip.to_dict() for strip in self.web_strips],
            "queries": list(self.web_queries),
            "source_urls": list(self.source_urls),
            "status": self.context.status,
            "context_text": self.context.text,
            "selected_strips": [strip.to_dict() for strip in self.context.strips],
            "citations": [citation.to_dict() for citation in self.context.citations],
            "answer": self.answer.to_dict() if self.answer else None,
            "warnings": list(self.warnings),
        }


class CragWorkflow:
    """LangGraph routing with one SQLite checkpoint thread per question/run."""

    def __init__(
        self, pipeline: IngestionPipeline, *, checkpoint_path: str | Path | None = None
    ) -> None:
        self.pipeline = pipeline
        self.checkpoint_path = Path(
            checkpoint_path or pipeline.config.checkpoint_path
        ).expanduser().resolve()

    def run(
        self,
        question: str,
        *,
        candidate_limit: int = 30,
        evaluation_limit: int = 10,
        document_id: str | None = None,
    ) -> CragRunResult:
        if not question.strip():
            raise ValueError("CRAG workflow requires a nonempty question")
        if candidate_limit <= 0 or not 0 < evaluation_limit <= candidate_limit:
            raise ValueError("Retrieval limits must be positive and evaluation <= candidate")
        run_id = uuid.uuid4().hex
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.checkpoint_path), check_same_thread=False)) as conn:
            saver = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
            graph = self._build_graph(saver)
            state = graph.invoke({
                "question": question,
                "candidate_limit": candidate_limit,
                "evaluation_limit": evaluation_limit,
                "document_id": document_id,
                "internal_strips": [],
                "web_strips": [],
                "web_queries": [],
                "source_urls": [],
                "warnings": [],
                "answer_pending": True,
            }, {"configurable": {"thread_id": run_id}})
        return self._result(run_id, state)

    def get_run(self, run_id: str) -> CragRunResult:
        """Read a completed run's persisted state without rerunning retrieval or APIs."""
        return self.load_run(self.checkpoint_path, run_id)

    @staticmethod
    def load_run(checkpoint_path: str | Path, run_id: str) -> CragRunResult:
        """Load a checkpoint without constructing Qdrant or embedding models."""
        path = Path(checkpoint_path).expanduser().resolve()
        if not run_id.strip() or not path.is_file():
            raise ValueError("CRAG checkpoint run was not found")
        with closing(sqlite3.connect(str(path), check_same_thread=False)) as conn:
            saver = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
            checkpoint = saver.get_tuple({"configurable": {"thread_id": run_id}})
            state = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
        if not state or "context" not in state or state.get("answer_pending"):
            raise ValueError("CRAG run is incomplete or was not found")
        return CragWorkflow._result_from_state(str(path), run_id, state)

    def _build_graph(self, saver: SqliteSaver) -> Any:
        builder = StateGraph(_GraphState)
        builder.add_node("evaluate", self._evaluate)
        builder.add_node("internal", self._internal)
        builder.add_node("web", self._web)
        builder.add_node("assemble", self._assemble)
        builder.add_node("generate", self._generate)
        builder.add_edge(START, "evaluate")
        builder.add_conditional_edges("evaluate", self._after_evaluate, {
            "internal": "internal", "web": "web",
        })
        builder.add_conditional_edges("internal", self._after_internal, {
            "web": "web", "assemble": "assemble",
        })
        builder.add_edge("web", "assemble")
        builder.add_edge("assemble", "generate")
        builder.add_edge("generate", END)
        return builder.compile(checkpointer=saver)

    def _evaluate(self, state: _GraphState) -> dict[str, Any]:
        decision = self.pipeline.evaluate_retrieval(
            state["question"],
            candidate_limit=state["candidate_limit"],
            evaluation_limit=state["evaluation_limit"],
            document_id=state["document_id"],
        )
        return {"decision": decision.to_dict()}

    @staticmethod
    def _after_evaluate(state: _GraphState) -> str:
        action = CorrectiveAction(state["decision"]["action"])
        return "web" if action == CorrectiveAction.INCORRECT else "internal"

    def _internal(self, state: _GraphState) -> dict[str, Any]:
        decision = _decision_from_dict(state["decision"])
        strips = self.pipeline.refine_internal_knowledge(state["question"], decision)
        return {"internal_strips": [strip.to_dict() for strip in strips]}

    @staticmethod
    def _after_internal(state: _GraphState) -> str:
        action = CorrectiveAction(state["decision"]["action"])
        return "web" if action == CorrectiveAction.AMBIGUOUS else "assemble"

    def _web(self, state: _GraphState) -> dict[str, Any]:
        decision = _decision_from_dict(state["decision"])
        found = self.pipeline.search_web_knowledge(state["question"], decision)
        return {
            "web_strips": [strip.to_dict() for strip in found.strips],
            "web_queries": list(found.queries),
            "source_urls": list(found.source_urls),
            "warnings": list(found.warnings),
        }

    def _assemble(self, state: _GraphState) -> dict[str, Any]:
        action = CorrectiveAction(state["decision"]["action"])
        assembled = ContextAssembler(
            self.pipeline.select_diverse_context, self.pipeline.config.context,
        ).assemble(
            state["question"], action,
            [_strip_from_dict(item) for item in state.get("internal_strips", [])],
            [_strip_from_dict(item) for item in state.get("web_strips", [])],
        )
        return {
            "context": assembled.to_dict(),
            "warnings": [*state.get("warnings", []), *assembled.warnings],
        }

    def _generate(self, state: _GraphState) -> dict[str, Any]:
        answer = self.pipeline.generate_answer(
            state["question"], _context_from_dict(state["context"])
        )
        return {"answer": answer.to_dict(), "answer_pending": False}

    def _result(self, run_id: str, state: _GraphState) -> CragRunResult:
        return self._result_from_state(str(self.checkpoint_path), run_id, state)

    @staticmethod
    def _result_from_state(
        checkpoint_path: str, run_id: str, state: _GraphState
    ) -> CragRunResult:
        return CragRunResult(
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            decision=_decision_from_dict(state["decision"]),
            internal_strips=tuple(_strip_from_dict(item) for item in state["internal_strips"]),
            web_strips=tuple(_strip_from_dict(item) for item in state["web_strips"]),
            web_queries=tuple(state["web_queries"]),
            source_urls=tuple(state["source_urls"]),
            context=_context_from_dict(state["context"]),
            answer=(
                GeneratedAnswer(
                    state["answer"]["status"], state["answer"]["text"],
                    state["answer"]["model"],
                    tuple(Citation(
                        item["marker"], item["strip_id"], item["source_type"],
                        item["source_ref"], item["metadata"],
                    ) for item in state["answer"]["citations"]),
                ) if "answer" in state else None
            ),
            warnings=tuple(state["warnings"]),
        )
