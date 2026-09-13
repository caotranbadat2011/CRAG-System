from __future__ import annotations

import math
from pathlib import Path

import pytest

from crag_ingestion.config import IngestionConfig
from crag_ingestion.embeddings.base import Embedder, EmbeddingVector
from crag_ingestion.models import Chunk, SearchResult
from crag_ingestion.pipeline import IngestionPipeline
from crag_ingestion.retrieval import (
    BGEReranker, KnowledgeStrip, RerankedRetriever, SemanticDiversityFilter,
)


class _FakeCrossEncoder:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[tuple[list[list[str]], bool]] = []

    def compute_score(self, pairs: list[list[str]], *, normalize: bool) -> object:
        self.calls.append((pairs, normalize))
        return self.scores


def _hit(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        score=score,
        chunk=Chunk(chunk_id, "doc", 0, text, len(text)),
        source_path="/source.txt",
        stored_path="/raw.txt",
    )


def test_bge_reranker_scores_pairs_without_probability_normalization() -> None:
    model = _FakeCrossEncoder([-1.2, 3.4])
    reranker = BGEReranker(model=model)
    assert reranker.name == "BAAI/bge-reranker-v2-m3"
    assert reranker.score("refund?", ["unrelated", "refund policy"]) == [-1.2, 3.4]
    assert model.calls == [
        ([ ["refund?", "unrelated"], ["refund?", "refund policy"] ], False)
    ]
    assert reranker.score("refund?", []) == []


def test_bge_reranker_accepts_single_scalar_and_rejects_bad_scores() -> None:
    assert BGEReranker(model=_FakeCrossEncoder(2.5)).score("q", ["doc"]) == [2.5]
    with pytest.raises(ValueError, match="inconsistent scores"):
        BGEReranker(model=_FakeCrossEncoder([1.0])).score("q", ["a", "b"])
    with pytest.raises(ValueError, match="invalid or inconsistent scores"):
        BGEReranker(model=_FakeCrossEncoder([math.nan])).score("q", ["doc"])


def test_retriever_reranks_hybrid_candidates_without_losing_rrf_score() -> None:
    calls: list[tuple[str, int, str | None]] = []

    def search(query: str, *, limit: int, document_id: str | None) -> list[SearchResult]:
        calls.append((query, limit, document_id))
        return [_hit("a", "first", 0.9), _hit("b", "second", 0.6)]

    model = _FakeCrossEncoder([-1.0, 4.0])
    retriever = RerankedRetriever(search, BGEReranker(model=model))
    ranked = retriever.retrieve("question", candidate_limit=20, evaluation_limit=2, document_id="doc")
    assert [item.result.chunk.chunk_id for item in ranked] == ["b", "a"]
    assert ranked[0].rerank_score == 4.0
    assert ranked[0].retrieval_score == 0.6
    assert ranked[0].to_dict()["retrieval_score"] == 0.6
    assert calls == [("question", 20, "doc")]
    with pytest.raises(ValueError, match="cannot exceed"):
        retriever.retrieve("question", candidate_limit=1, evaluation_limit=2)


class _VectorEmbedder(Embedder):
    name = "test/vector"
    dimensions = 2

    _vectors = {
        "query": [1.0, 0.0],
        "answer": [1.0, 0.0],
        "duplicate": [0.999, 0.001],
        "complement": [0.8, 0.6],
        "irrelevant": [0.0, 1.0],
    }

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector(self._vectors[text], {1: 1.0}) for text in texts]


def _strip(strip_id: str, text: str, source_type: str) -> KnowledgeStrip:
    return KnowledgeStrip(strip_id, text, source_type, f"{source_type}:{strip_id}")


def test_semantic_diversity_removes_near_duplicates_and_keeps_provenance() -> None:
    strips = [
        _strip("a", "answer", "internal"),
        _strip("b", "duplicate", "internal"),
        _strip("c", "complement", "web"),
        _strip("d", "irrelevant", "web"),
    ]
    selector = SemanticDiversityFilter(_VectorEmbedder())
    selected = selector.select("query", strips, limit=2)
    assert [item.strip_id for item in selected] == ["a", "c"]
    assert [item.source_ref for item in selected] == ["internal:a", "web:c"]
    assert selector.select(
        "query", strips, limit=2, min_per_source={"internal": 1, "web": 1}
    ) == selected


def test_semantic_diversity_mmr_prefers_new_information_when_weighted() -> None:
    strips = [
        _strip("a", "answer", "internal"),
        _strip("b", "duplicate", "internal"),
        _strip("c", "complement", "internal"),
    ]
    selector = SemanticDiversityFilter(
        _VectorEmbedder(), relevance_weight=0.3, duplicate_threshold=1.0
    )
    assert [item.strip_id for item in selector.select("query", strips, limit=2)] == [
        "a", "c"
    ]


def test_semantic_diversity_quotas_and_input_guards() -> None:
    selector = SemanticDiversityFilter(_VectorEmbedder())
    strips = [_strip("a", "answer", "internal"), _strip("b", "duplicate", "web")]
    with pytest.raises(ValueError, match="near-duplicate"):
        selector.select(
            "query", strips, limit=2, min_per_source={"internal": 1, "web": 1}
        )
    with pytest.raises(ValueError, match="Source quotas"):
        selector.select("query", strips, limit=1, min_per_source={"internal": 1, "web": 1})
    with pytest.raises(ValueError, match="unique"):
        selector.select("query", [strips[0], strips[0]], limit=2)


def test_pipeline_exposes_reranked_candidates_for_future_evaluator(
    tmp_path: Path, fake_embedder: Embedder,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Chính sách hoàn tiền trong vòng ba mươi ngày.", encoding="utf-8")
    model = _FakeCrossEncoder([2.0])
    config = IngestionConfig(data_dir=tmp_path / "data")
    with IngestionPipeline(
        config, embedder=fake_embedder, reranker=BGEReranker(model=model)
    ) as pipeline:
        pipeline.ingest_file(source)
        results = pipeline.retrieve_for_evaluation(
            "hoàn tiền", candidate_limit=5, evaluation_limit=1
        )
        assert len(results) == 1
        assert results[0].result.chunk.text == source.read_text(encoding="utf-8")
        assert results[0].rerank_score == 2.0
