from __future__ import annotations

import hashlib
import math
import re

import pytest

from crag_ingestion.embeddings.base import Embedder


class FakeSentenceTransformerEmbedder(Embedder):
    """Small deterministic test double; production always uses Sentence Transformers."""

    name = "test/sentence-transformer"
    dimensions = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[bucket] += 1.0 if digest[4] & 1 else -1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


@pytest.fixture
def fake_embedder() -> FakeSentenceTransformerEmbedder:
    return FakeSentenceTransformerEmbedder()
