from __future__ import annotations

import hashlib
import math
import re
import unicodedata

from .base import Embedder

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free baseline suitable for ingestion tests.

    It is lexical rather than semantic. Production can switch to the sentence
    transformer adapter without changing stored/indexed data contracts.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("Hashing embeddings require at least 32 dimensions")
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-v1-{self.dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens = _TOKEN.findall(normalized)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "little") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

