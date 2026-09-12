from __future__ import annotations

from ..config import IngestionConfig
from .base import Embedder
from .sentence_transformer import SentenceTransformerEmbedder


def create_embedder(config: IngestionConfig) -> Embedder:
    return SentenceTransformerEmbedder(config.embedding_model)
