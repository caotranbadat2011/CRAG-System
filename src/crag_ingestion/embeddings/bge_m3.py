from __future__ import annotations

import math
from typing import Any

from .base import Embedder, EmbeddingVector


class BGEM3Embedder(Embedder):
    """Extract BGE-M3 dense vectors and token-id lexical weights in one pass."""

    DIMENSIONS = 1024

    def __init__(self, model_name: str, *, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        self._ensure_model()
        output = self._model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        dense_rows = output["dense_vecs"]
        lexical_rows = output["lexical_weights"]
        if hasattr(dense_rows, "tolist"):
            dense_rows = dense_rows.tolist()
        if len(dense_rows) != len(texts) or len(lexical_rows) != len(texts):
            raise ValueError("BGE-M3 returned an inconsistent embedding count")

        vectors: list[EmbeddingVector] = []
        for dense_row, lexical_row in zip(dense_rows, lexical_rows, strict=True):
            dense = [float(value) for value in dense_row]
            if len(dense) != self.dimensions or not all(math.isfinite(v) for v in dense):
                raise ValueError("BGE-M3 returned invalid dense vector dimensions or values")
            norm = math.sqrt(sum(value * value for value in dense))
            if not norm or not math.isfinite(norm):
                raise ValueError("BGE-M3 returned a zero or invalid dense vector")
            dense = [value / norm for value in dense]

            if not isinstance(lexical_row, dict):
                raise ValueError("BGE-M3 returned invalid lexical_weights")
            weights: dict[int, float] = {}
            for token_id, raw_weight in lexical_row.items():
                token = int(token_id)
                weight = float(raw_weight)
                if token < 0 or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("BGE-M3 returned invalid lexical_weights")
                if token in weights:
                    raise ValueError("BGE-M3 returned duplicate lexical token IDs")
                weights[token] = weight
            vectors.append(EmbeddingVector(dense=dense, lexical_weights=weights))
        return vectors

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is required for BGE-M3 dense and sparse embeddings; "
                "reinstall the project dependencies"
            ) from exc
        self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
