from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from crag_ingestion.config import ContextConfig, IngestionConfig
from crag_ingestion.cli import main
from crag_ingestion.models import Chunk, SearchResult
from crag_ingestion.pipeline import IngestionPipeline
from crag_ingestion.retrieval import (
    CorrectiveAction, EvaluatedHit, EvaluationDecision, GeneratedAnswer,
    KnowledgeStrip, Relevance,
    RerankedHit, WebKnowledgeResult,
)
from crag_ingestion.retrieval.context import ContextAssembler
from crag_ingestion.workflow import CragWorkflow


def _internal() -> KnowledgeStrip:
    return KnowledgeStrip(
        "chunk-1:strip:0-27", "Hoàn tiền trong 30 ngày.", "internal", "chunk-1",
        {"chunk_id": "chunk-1", "source_path": "D:/docs/policy.pdf", "pages": [3],
         "chunk_char_start": 0, "chunk_char_end": 23},
    )


def _web() -> KnowledgeStrip:
    return KnowledgeStrip(
        "web:abc", "Chính sách hoàn tiền được công bố.", "web",
        "https://example.org/refunds",
        {"url": "https://example.org/refunds", "page_char_start": 10,
         "page_char_end": 43, "fetched_at": "2026-09-14T00:00:00Z"},
    )


def _decision(action: CorrectiveAction) -> EvaluationDecision:
    chunk = Chunk("chunk-1", "doc-1", 0, "Hoàn tiền trong 30 ngày.", 23,
                  pages=(3,), heading_path=("Chính sách",))
    search = SearchResult(0.7, chunk, "D:/docs/policy.pdf", "D:/data/raw/policy.pdf")
    label = {
        CorrectiveAction.CORRECT: Relevance.RELEVANT,
        CorrectiveAction.INCORRECT: Relevance.IRRELEVANT,
        CorrectiveAction.AMBIGUOUS: Relevance.UNCERTAIN,
    }[action]
    return EvaluationDecision(action, "test/evaluator", (
        EvaluatedHit(RerankedHit(search, 2.3), label, "checked"),
    ))


class FakePipeline:
    def __init__(self, tmp_path: Path, action: CorrectiveAction) -> None:
        self.config = IngestionConfig(data_dir=tmp_path / "runtime")
        self.action = action
        self.calls: list[str] = []
        self.selection_options: list[dict[str, object]] = []
        self.web_strips: list[KnowledgeStrip] = [_web()]

    def evaluate_retrieval(self, question: str, **kwargs: object) -> EvaluationDecision:
        assert question == "hoàn tiền"
        assert kwargs == {"candidate_limit": 7, "evaluation_limit": 3, "document_id": None}
        self.calls.append("evaluate")
        return _decision(self.action)

    def refine_internal_knowledge(
        self, question: str, decision: EvaluationDecision
    ) -> list[KnowledgeStrip]:
        assert question == "hoàn tiền"
        assert decision.action == self.action
        assert decision.hits[0].hit.result.chunk.pages == (3,)
        assert decision.hits[0].hit.result.chunk.heading_path == ("Chính sách",)
        self.calls.append("internal")
        return [_internal()]

    def search_web_knowledge(
        self, question: str, decision: EvaluationDecision
    ) -> WebKnowledgeResult:
        assert question == "hoàn tiền" and decision.action == self.action
        self.calls.append("web")
        return WebKnowledgeResult(
            self.action, ("refund policy",), ("https://example.org/refunds",),
            tuple(self.web_strips), (),
        )

    def select_diverse_context(
        self, question: str, strips: list[KnowledgeStrip], **kwargs: object
    ) -> list[KnowledgeStrip]:
        assert question == "hoàn tiền"
        self.calls.append("select")
        self.selection_options.append(kwargs)
        return strips[:int(kwargs["limit"])]

    def generate_answer(self, question: str, context: object) -> GeneratedAnswer:
        assert question == "hoàn tiền"
        self.calls.append("generate")
        if context.status == "no_evidence":  # type: ignore[attr-defined]
            return GeneratedAnswer("insufficient_evidence", "Không đủ bằng chứng.", None, ())
        citations = context.citations  # type: ignore[attr-defined]
        return GeneratedAnswer(
            "partial" if context.status == "partial" else "answered",  # type: ignore[attr-defined]
            f"Đã có bằng chứng. {citations[0].marker}", "test/generator", (citations[0],),
        )


@pytest.mark.parametrize("action,expected_calls,expected_sources", [
    (CorrectiveAction.CORRECT, ["evaluate", "internal", "select", "generate"], ["internal"]),
    (CorrectiveAction.INCORRECT, ["evaluate", "web", "select", "generate"], ["web"]),
    (CorrectiveAction.AMBIGUOUS, ["evaluate", "internal", "web", "select", "generate"], ["internal", "web"]),
])
def test_graph_routes_branches_and_persists_cited_context(
    tmp_path: Path, action: CorrectiveAction, expected_calls: list[str],
    expected_sources: list[str],
) -> None:
    pipeline = FakePipeline(tmp_path, action)
    workflow = CragWorkflow(pipeline)  # type: ignore[arg-type]
    result = workflow.run("hoàn tiền", candidate_limit=7, evaluation_limit=3)

    assert pipeline.calls == expected_calls
    assert result.context.status == "ready"
    assert result.answer is not None and result.answer.status == "answered"
    assert result.answer.citations[0].marker == "[1]"
    assert [strip.source_type for strip in result.context.strips] == expected_sources
    assert [item.marker for item in result.context.citations] == [
        f"[{index}]" for index in range(1, len(expected_sources) + 1)
    ]
    assert [json.loads(line)["citation"] for line in result.context.text.splitlines()] == [
        item.marker for item in result.context.citations
    ]
    assert len(result.context.text) <= pipeline.config.context.max_chars
    assert result.context.citations[0].metadata
    assert pipeline.selection_options[0]["relevance_weight"] == 0.6
    assert pipeline.selection_options[0]["duplicate_threshold"] == 0.85
    assert pipeline.selection_options[0]["min_per_source"] == (
        {"internal": 1, "web": 1} if action == CorrectiveAction.AMBIGUOUS else None
    )
    assert Path(result.checkpoint_path).is_file()
    assert CragWorkflow.load_run(result.checkpoint_path, result.run_id).to_dict() == result.to_dict()
    assert pipeline.calls == expected_calls  # checkpoint read must not call nodes
    with sqlite3.connect(result.checkpoint_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (result.run_id,)
        ).fetchone()[0] >= 3


def test_missing_web_evidence_is_not_replaced_by_search_snippets(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path, CorrectiveAction.INCORRECT)
    pipeline.web_strips = []
    result = CragWorkflow(pipeline).run(  # type: ignore[arg-type]
        "hoàn tiền", candidate_limit=7, evaluation_limit=3,
    )
    assert result.context.status == "no_evidence"
    assert result.context.text == "" and result.context.citations == ()
    assert "No relevant evidence was found" in result.warnings
    assert pipeline.calls == ["evaluate", "web", "generate"]
    assert result.answer is not None and result.answer.status == "insufficient_evidence"


def test_ambiguous_with_one_source_is_partial(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path, CorrectiveAction.AMBIGUOUS)
    pipeline.web_strips = []
    result = CragWorkflow(pipeline).run(  # type: ignore[arg-type]
        "hoàn tiền", candidate_limit=7, evaluation_limit=3,
    )
    assert result.context.status == "partial"
    assert result.answer is not None and result.answer.status == "partial"
    assert [item.source_type for item in result.context.citations] == ["internal"]
    assert any("one source" in warning for warning in result.warnings)


def test_context_budget_preserves_exact_strip_and_citation() -> None:
    assembler = ContextAssembler(
        lambda _q, strips, **_kw: strips,
        ContextConfig(max_strips=2, max_chars=160),
    )
    result = assembler.assemble(
        "hoàn tiền", CorrectiveAction.AMBIGUOUS, [_internal()], [_web()],
    )
    assert len(result.text) <= 160
    assert len(result.strips) == len(result.citations) == 1
    assert json.loads(result.text)["text"] == result.strips[0].text
    assert result.status == "partial"
    assert any("budget" in warning for warning in result.warnings)


def test_quota_conflict_falls_back_with_warning() -> None:
    calls: list[dict[str, object]] = []

    def selector(_question: str, strips: list[KnowledgeStrip], **kwargs: object) -> list[KnowledgeStrip]:
        calls.append(kwargs)
        if kwargs["min_per_source"]:
            raise ValueError("Cannot meet web quota without near-duplicate strips")
        return strips[:1]

    context = ContextAssembler(selector).assemble(
        "hoàn tiền", CorrectiveAction.AMBIGUOUS, [_internal()], [_web()],
    )
    assert len(calls) == 2 and calls[1]["min_per_source"] is None
    assert context.status == "partial"
    assert any("near-duplicate" in warning for warning in context.warnings)


def test_context_rejects_wrong_branch_and_duplicate_ids() -> None:
    assembler = ContextAssembler(lambda _q, strips, **_kw: strips)
    with pytest.raises(ValueError, match="Correct branch"):
        assembler.assemble("hoàn tiền", CorrectiveAction.CORRECT, [_internal()], [_web()])
    duplicate = KnowledgeStrip(_internal().strip_id, "another", "web", "https://example.org")
    with pytest.raises(ValueError, match="unique"):
        assembler.assemble("hoàn tiền", CorrectiveAction.AMBIGUOUS, [_internal()], [duplicate])


def test_context_rejects_selector_that_exceeds_limit() -> None:
    assembler = ContextAssembler(
        lambda _q, strips, **_kw: strips,
        ContextConfig(max_strips=1),
    )
    with pytest.raises(ValueError, match="too much"):
        assembler.assemble("hoàn tiền", CorrectiveAction.AMBIGUOUS, [_internal()], [_web()])


def test_load_unknown_run_and_invalid_input(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path, CorrectiveAction.CORRECT)
    workflow = CragWorkflow(pipeline)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        workflow.run(" ")
    with pytest.raises(ValueError, match="limits"):
        workflow.run("hoàn tiền", candidate_limit=2, evaluation_limit=3)
    with pytest.raises(ValueError, match="not found"):
        workflow.get_run("missing")


def test_legacy_context_only_state_remains_readable(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path, CorrectiveAction.CORRECT)
    workflow = CragWorkflow(pipeline)  # type: ignore[arg-type]
    result = workflow.run("hoàn tiền", candidate_limit=7, evaluation_limit=3)
    with sqlite3.connect(result.checkpoint_path) as conn:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        saver = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        checkpoint = saver.get_tuple({"configurable": {"thread_id": result.run_id}})
        assert checkpoint is not None
        state = dict(checkpoint.checkpoint["channel_values"])
    state.pop("answer")
    state.pop("answer_pending")
    legacy = CragWorkflow._result_from_state(result.checkpoint_path, result.run_id, state)
    assert legacy.answer is None
    assert legacy.context.status == "ready"


def test_failed_answer_generation_does_not_expose_incomplete_run(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path, CorrectiveAction.CORRECT)

    def fail_answer(_question: str, _context: object) -> GeneratedAnswer:
        raise ValueError("Gemini answer contains an unknown citation")

    pipeline.generate_answer = fail_answer  # type: ignore[method-assign]
    workflow = CragWorkflow(pipeline)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown citation"):
        workflow.run("hoàn tiền", candidate_limit=7, evaluation_limit=3)
    with sqlite3.connect(workflow.checkpoint_path) as conn:
        run_id = conn.execute(
            "SELECT thread_id FROM checkpoints ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    with pytest.raises(ValueError, match="incomplete"):
        workflow.get_run(run_id)


def test_cli_run_passes_context_limits_and_routing_args(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeCliPipeline:
        def __init__(self, config: IngestionConfig) -> None:
            assert config.context.max_strips == 4
            assert config.context.max_chars == 2000

        def __enter__(self) -> "FakeCliPipeline":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def run_crag(self, question: str, **kwargs: object) -> object:
            assert question == "hoàn tiền"
            assert kwargs == {
                "candidate_limit": 9, "evaluation_limit": 3, "document_id": "doc-1",
            }
            return SimpleNamespace(to_dict=lambda: {"status": "ready", "citations": []})

    monkeypatch.setattr("crag_ingestion.cli.IngestionPipeline", FakeCliPipeline)
    assert main([
        "--context-max-strips", "4", "--context-max-chars", "2000",
        "run", "hoàn tiền", "--candidate-limit", "9", "--limit", "3",
        "--document-id", "doc-1",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_cli_show_run_reads_sqlite_without_loading_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = FakePipeline(tmp_path, CorrectiveAction.CORRECT)
    result = CragWorkflow(pipeline).run(  # type: ignore[arg-type]
        "hoàn tiền", candidate_limit=7, evaluation_limit=3,
    )

    def fail_pipeline(_config: IngestionConfig) -> object:
        raise AssertionError("show-run must not initialize Qdrant or BGE")

    monkeypatch.setattr("crag_ingestion.cli.IngestionPipeline", fail_pipeline)
    assert main([
        "--data-dir", str(tmp_path / "runtime"), "show-run", result.run_id,
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == result.to_dict()


def test_pipeline_run_crag_uses_real_diversity_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_embedder: object,
) -> None:
    config = IngestionConfig(data_dir=tmp_path / "runtime")
    class FakeAnswerGenerator:
        def generate(self, question: str, context: object) -> GeneratedAnswer:
            assert question == "hoàn tiền"
            return GeneratedAnswer(
                "answered", f"Đã có nguồn. {context.citations[0].marker}",  # type: ignore[attr-defined]
                "test/generator", (context.citations[0],),  # type: ignore[attr-defined]
            )

    with IngestionPipeline(
        config, embedder=fake_embedder, answer_generator=FakeAnswerGenerator()  # type: ignore[arg-type]
    ) as pipeline:  # type: ignore[arg-type]
        monkeypatch.setattr(
            pipeline, "evaluate_retrieval",
            lambda _question, **_kwargs: _decision(CorrectiveAction.AMBIGUOUS),
        )
        monkeypatch.setattr(
            pipeline, "refine_internal_knowledge", lambda _question, _decision: [_internal()],
        )
        monkeypatch.setattr(
            pipeline, "search_web_knowledge",
            lambda _question, _decision: WebKnowledgeResult(
                CorrectiveAction.AMBIGUOUS, ("refund policy",),
                ("https://example.org/refunds",), (_web(),), (),
            ),
        )
        result = pipeline.run_crag("hoàn tiền", candidate_limit=7, evaluation_limit=3)
    assert result.context.status == "ready"
    assert result.answer is not None and result.answer.status == "answered"
    assert {item.source_type for item in result.context.citations} == {"internal", "web"}
