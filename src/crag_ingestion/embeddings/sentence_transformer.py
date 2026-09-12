from __future__ import annotations

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'semantic' extra to use sentence-transformer embeddings"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        values = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return values.tolist()

