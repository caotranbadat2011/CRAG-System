from __future__ import annotations

import pytest

from crag_ingestion.config import IngestionConfig
from crag_ingestion.embeddings import SentenceTransformerEmbedder
from crag_ingestion.embeddings import factory


class _FakeModel:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.calls: list[tuple[list[str], bool, bool]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimensions

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append((texts, normalize_embeddings, show_progress_bar))
        return [[1, 0.5, 0] for _ in texts]


def test_sentence_transformer_embedder_uses_normalized_model_vectors() -> None:
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder("test/model", model=model)

    assert embedder.name == "test/model"
    assert embedder.dimensions == 3
    assert embedder.embed(["xin chào", "tài liệu"]) == [
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
    ]
    assert model.calls == [(["xin chào", "tài liệu"], True, False)]
    assert embedder.embed([]) == []


def test_sentence_transformer_embedder_rejects_invalid_dimensions() -> None:
    embedder = SentenceTransformerEmbedder("test/model", model=_FakeModel(0))
    with pytest.raises(ValueError, match="invalid embedding dimension"):
        _ = embedder.dimensions


def test_sentence_transformer_embedder_rejects_inconsistent_vectors() -> None:
    model = _FakeModel(4)
    embedder = SentenceTransformerEmbedder("test/model", model=model)
    with pytest.raises(ValueError, match="inconsistent vector dimensions"):
        embedder.embed(["text"])


def test_factory_always_builds_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(factory, "SentenceTransformerEmbedder", lambda model_name: sentinel)
    config = IngestionConfig(embedding_model="custom/model")

    assert factory.create_embedder(config) is sentinel
