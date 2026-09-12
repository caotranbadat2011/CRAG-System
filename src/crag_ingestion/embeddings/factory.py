from __future__ import annotations

from ..config import IngestionConfig
from .base import Embedder
from .hashing import HashingEmbedder


def create_embedder(config: IngestionConfig) -> Embedder:
    if config.embedding_provider == "hash":
        return HashingEmbedder(config.embedding_dimensions)
    if config.embedding_provider == "sentence-transformers":
        from .sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(config.embedding_model)
    raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")

