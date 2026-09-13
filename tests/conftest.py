from __future__ import annotations

import hashlib
import math
import re

import pytest

from crag_ingestion.embeddings.base import Embedder, EmbeddingVector


class FakeBGEM3Embedder(Embedder):
    """Small deterministic dense/sparse test double without model downloads."""

    name = "test/bge-m3"
    dimensions = 64

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        vectors: list[EmbeddingVector] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            lexical_weights: dict[int, float] = {}
            for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[bucket] += 1.0 if digest[4] & 1 else -1.0
                token_id = int.from_bytes(digest[:4], "big")
                lexical_weights[token_id] = lexical_weights.get(token_id, 0.0) + 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append(EmbeddingVector(
                dense=[value / norm for value in vector] if norm else vector,
                lexical_weights=lexical_weights,
            ))
        return vectors


@pytest.fixture
def fake_embedder() -> FakeBGEM3Embedder:
    return FakeBGEM3Embedder()
