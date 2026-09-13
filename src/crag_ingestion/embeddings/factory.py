from __future__ import annotations

from ..config import IngestionConfig
from .base import Embedder
from .bge_m3 import BGEM3Embedder


def create_embedder(config: IngestionConfig) -> Embedder:
    return BGEM3Embedder(config.embedding_model)
