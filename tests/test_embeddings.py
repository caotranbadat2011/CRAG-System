from __future__ import annotations

import pytest

from crag_ingestion.config import IngestionConfig
from crag_ingestion.embeddings import BGEM3Embedder
from crag_ingestion.embeddings import factory


class _FakeModel:
    def __init__(self, *, lexical_weights: list[dict[str, float]] | None = None) -> None:
        self.calls: list[tuple[list[str], bool, bool, bool]] = []
        self.lexical_weights = lexical_weights

    def encode(
        self, texts: list[str], *, return_dense: bool, return_sparse: bool,
        return_colbert_vecs: bool,
    ) -> dict[str, object]:
        self.calls.append((texts, return_dense, return_sparse, return_colbert_vecs))
        return {
            "dense_vecs": [[3.0, 4.0] + [0.0] * 1022 for _ in texts],
            "lexical_weights": self.lexical_weights or [{"42": 0.75, "7": 1.25} for _ in texts],
        }


def test_bge_m3_extracts_dense_and_lexical_weights_once() -> None:
    model = _FakeModel()
    embedder = BGEM3Embedder("BAAI/bge-m3", model=model)

    assert embedder.name == "BAAI/bge-m3"
    assert embedder.dimensions == 1024
    vectors = embedder.embed(["xin chào", "tài liệu"])
    assert len(vectors) == 2
    assert vectors[0].dense[:2] == [0.6, 0.8]
    assert vectors[0].lexical_weights == {42: 0.75, 7: 1.25}
    assert model.calls == [(["xin chào", "tài liệu"], True, True, False)]
    assert embedder.embed([]) == []


def test_bge_m3_rejects_bad_dense_shape() -> None:
    class BadShape(_FakeModel):
        def encode(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {"dense_vecs": [[1.0]], "lexical_weights": [{"1": 1.0}]}

    with pytest.raises(ValueError, match="dense vector dimensions"):
        BGEM3Embedder("BAAI/bge-m3", model=BadShape()).embed(["text"])


def test_bge_m3_rejects_invalid_lexical_weights() -> None:
    with pytest.raises(ValueError, match="lexical_weights"):
        BGEM3Embedder("BAAI/bge-m3", model=_FakeModel(lexical_weights=[{"-1": 1.0}])).embed(["text"])


def test_factory_builds_single_bge_m3_model(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(factory, "BGEM3Embedder", lambda model_name: sentinel)
    assert factory.create_embedder(IngestionConfig()) is sentinel


def test_default_embedding_model_and_collection() -> None:
    config = IngestionConfig()
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.qdrant_collection == "crag_bge_m3_hybrid"
