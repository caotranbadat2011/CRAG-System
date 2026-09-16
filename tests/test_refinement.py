from __future__ import annotations

from pathlib import Path

import pytest

from crag_ingestion.config import IngestionConfig, RefinementConfig
from crag_ingestion.embeddings.base import Embedder
from crag_ingestion.models import Chunk, SearchResult
from crag_ingestion.pipeline import IngestionPipeline
from crag_ingestion.retrieval import (
    CorrectiveAction, EvaluatedHit, EvaluationDecision, KnowledgeRefiner,
    Relevance, RerankedHit,
)


def _parent(chunk_id: str, text: str, relevance: Relevance) -> EvaluatedHit:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="document-1",
        ordinal=2,
        text=text,
        char_count=len(text),
        pages=(3,),
        heading_path=("Chính sách",),
        metadata={
            "source_metadata": [{"source": {"path": "/original.pdf", "page": 3}}],
            "has_overlap": True,
        },
    )
    hit = RerankedHit(SearchResult(0.72, chunk, "/original.pdf", "/raw.pdf"), 3.4)
    return EvaluatedHit(hit, relevance, "Parent judgment")


class FakeStripEvaluator:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def evaluate(self, query: str, hits: list[RerankedHit]) -> EvaluationDecision:
        assert query == "hoàn tiền"
        self.calls.append([hit.result.chunk.chunk_id for hit in hits])
        judged = tuple(
            EvaluatedHit(
                hit,
                Relevance.RELEVANT if "hoàn tiền" in hit.result.chunk.text.lower() else Relevance.IRRELEVANT,
                "Strip checked",
            )
            for hit in hits
        )
        return EvaluationDecision(CorrectiveAction.CORRECT, "fake/strip", judged)


def test_correct_refines_only_relevant_parent_and_retains_exact_chunk_offsets() -> None:
    text = "Có thể hoàn tiền trong 30 ngày. Nội dung này nói về vận chuyển."
    judge = FakeStripEvaluator()
    refiner = KnowledgeRefiner(judge, RefinementConfig(80, 1, 2))
    decision = EvaluationDecision(CorrectiveAction.CORRECT, "fake/parent", (
        _parent("keep", text, Relevance.RELEVANT),
        _parent("drop", "Có thể hoàn tiền trong 60 ngày.", Relevance.IRRELEVANT),
    ))

    strips = refiner.refine_internal("hoàn tiền", decision)

    assert len(strips) == 1
    strip = strips[0]
    assert strip.text == "Có thể hoàn tiền trong 30 ngày."
    assert strip.source_type == "internal"
    assert strip.source_ref == "keep"
    assert strip.metadata["chunk_id"] == "keep"
    assert strip.metadata["document_id"] == "document-1"
    assert strip.metadata["source_path"] == "/original.pdf"
    assert strip.metadata["stored_path"] == "/raw.pdf"
    assert strip.metadata["pages"] == [3]
    assert strip.metadata["heading_path"] == ["Chính sách"]
    assert strip.metadata["source_metadata"] == [
        {"source": {"path": "/original.pdf", "page": 3}}
    ]
    assert strip.metadata["strip_reason"] == "Strip checked"
    assert text[strip.metadata["chunk_char_start"]:strip.metadata["chunk_char_end"]] == strip.text
    assert all(not item.startswith("drop:") for call in judge.calls for item in call)


def test_ambiguous_refines_uncertain_internal_parent_but_not_irrelevant_parent() -> None:
    judge = FakeStripEvaluator()
    decision = EvaluationDecision(CorrectiveAction.AMBIGUOUS, "fake/parent", (
        _parent("uncertain", "Quy định hoàn tiền chưa đầy đủ.", Relevance.UNCERTAIN),
        _parent("irrelevant", "Không liên quan.", Relevance.IRRELEVANT),
    ))
    strips = KnowledgeRefiner(judge).refine_internal("hoàn tiền", decision)
    assert [strip.source_ref for strip in strips] == ["uncertain"]
    assert strips[0].metadata["parent_relevance"] == "uncertain"


def test_incorrect_branch_produces_no_internal_strips_or_api_calls() -> None:
    judge = FakeStripEvaluator()
    decision = EvaluationDecision(CorrectiveAction.INCORRECT, "fake/parent", (
        _parent("bad", "Có thể hoàn tiền.", Relevance.IRRELEVANT),
    ))
    assert KnowledgeRefiner(judge).refine_internal("hoàn tiền", decision) == []
    assert judge.calls == []


def test_strip_judgments_are_batched_and_preserve_source_order() -> None:
    judge = FakeStripEvaluator()
    text = "Hoàn tiền sau một ngày. Không liên quan. Hoàn tiền sau hai ngày."
    decision = EvaluationDecision(CorrectiveAction.CORRECT, "fake/parent", (
        _parent("parent", text, Relevance.RELEVANT),
    ))
    strips = KnowledgeRefiner(judge, RefinementConfig(80, 1, 2)).refine_internal(
        "hoàn tiền", decision
    )
    assert [strip.text for strip in strips] == [
        "Hoàn tiền sau một ngày.", "Hoàn tiền sau hai ngày."
    ]
    assert [len(call) for call in judge.calls] == [2, 1]


def test_long_unbroken_text_is_split_without_exceeding_limit_or_losing_offsets() -> None:
    text = "đ" * 195
    refiner = KnowledgeRefiner(FakeStripEvaluator(), RefinementConfig(80, 2, 12))
    spans = refiner._spans(text)
    assert [end - start for start, end in spans] == [80, 80, 35]
    assert "".join(text[start:end] for start, end in spans) == text


def test_pdf_line_break_does_not_separate_negation_from_predicate() -> None:
    text = (
        "Hai ràng buộc: 31 . 0% câu có ngữ cảnh không\n\n"
        "nhét vừa một cửa sổ 256 token, và 32 . 4% câu là câu bẫy, "
        "chỉ khác một câu hỏi thật ở\n\nmột chi tiết then chốt đã bị thay. "
        "Hệ thống phải biết từ chối."
    )
    spans = KnowledgeRefiner(FakeStripEvaluator()).split_spans(text)
    strips = [text[start:end] for start, end in spans]
    assert any("31 . 0%" in strip and "không\n\nnhét vừa" in strip for strip in strips)
    assert not any(strip.startswith("nhét vừa") or strip.endswith("31 .") for strip in strips)


def test_duplicate_parent_ids_are_rejected() -> None:
    judge = FakeStripEvaluator()
    parent = _parent("same", "Hoàn tiền.", Relevance.RELEVANT)
    decision = EvaluationDecision(CorrectiveAction.CORRECT, "fake/parent", (parent, parent))
    try:
        KnowledgeRefiner(judge).refine_internal("hoàn tiền", decision)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate parent IDs must fail")


def test_missing_strip_judgment_fails_closed() -> None:
    class IncompleteEvaluator:
        def evaluate(self, _query: str, _hits: list[RerankedHit]) -> EvaluationDecision:
            return EvaluationDecision(CorrectiveAction.INCORRECT, "fake", ())

    decision = EvaluationDecision(CorrectiveAction.CORRECT, "fake/parent", (
        _parent("parent", "Hoàn tiền trong 30 ngày.", Relevance.RELEVANT),
    ))
    with pytest.raises(ValueError, match="incomplete"):
        KnowledgeRefiner(IncompleteEvaluator()).refine_internal("hoàn tiền", decision)


def test_pipeline_exposes_refinement_for_correct_branch(
    tmp_path: Path, fake_embedder: Embedder,
) -> None:
    judge = FakeStripEvaluator()
    decision = EvaluationDecision(CorrectiveAction.CORRECT, "fake/parent", (
        _parent("parent", "Hoàn tiền trong 30 ngày.", Relevance.RELEVANT),
    ))
    with IngestionPipeline(
        IngestionConfig(data_dir=tmp_path / "runtime"),
        embedder=fake_embedder,
        evaluator=judge,
    ) as pipeline:
        strips = pipeline.refine_internal_knowledge("hoàn tiền", decision)
    assert [strip.text for strip in strips] == ["Hoàn tiền trong 30 ngày."]
    assert len(judge.calls) == 1
