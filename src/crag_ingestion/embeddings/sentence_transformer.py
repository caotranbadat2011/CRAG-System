from __future__ import annotations

from typing import Any

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, *, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dimensions(self) -> int:
        self._ensure_model()
        assert self._dimensions is not None
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_model()
        values = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        rows = values.tolist() if hasattr(values, "tolist") else values
        vectors = [[float(value) for value in row] for row in rows]
        if len(vectors) != len(texts):
            raise ValueError("Sentence Transformer model returned an inconsistent vector count")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("Sentence Transformer model returned inconsistent vector dimensions")
        return vectors

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for CRAG embeddings; reinstall the project dependencies"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        if self._dimensions is None:
            self._dimensions = int(self._model.get_sentence_embedding_dimension())
            if self._dimensions <= 0:
                raise ValueError("Sentence Transformer model returned an invalid embedding dimension")
