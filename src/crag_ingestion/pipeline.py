from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .chunking import StructuralChunker
from .cleaning import DocumentCleaner
from .config import IngestionConfig
from .embeddings import Embedder, create_embedder
from .exceptions import EmptyDocumentError, FileTooLargeError, IngestionError
from .index import SQLiteVectorIndex
from .models import IngestionResult, SearchResult
from .parsers import ParserRegistry, default_registry
from .storage import ArtifactStore
from .utils import canonical_json, sha256_file, stable_document_id
from .validation import ValidationReport, validate_index


class IngestionPipeline:
    # Bump whenever parsing semantics change so existing documents are rebuilt.
    PIPELINE_VERSION = "2"

    def __init__(
        self,
        config: IngestionConfig | None = None,
        *,
        registry: ParserRegistry | None = None,
        embedder: Embedder | None = None,
        index: SQLiteVectorIndex | None = None,
    ) -> None:
        self.config = config or IngestionConfig()
        self.registry = registry or default_registry(self.config)
        self.embedder = embedder or create_embedder(self.config)
        self.cleaner = DocumentCleaner(self.config.cleaning)
        self.chunker = StructuralChunker(self.config.chunking)
        self.artifacts = ArtifactStore(self.config.raw_dir, self.config.processed_dir)
        self.index = index or SQLiteVectorIndex(self.config.index_path)
        self._owns_index = index is None

    @property
    def pipeline_signature(self) -> str:
        settings = {
            "version": self.PIPELINE_VERSION,
            "cleaning": asdict(self.config.cleaning),
            "chunking": asdict(self.config.chunking),
            "embedder": self.embedder.name,
            "dimensions": self.embedder.dimensions,
            "parsers": self.registry.supported_extensions,
        }
        return hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()

    def close(self) -> None:
        if self._owns_index:
            self.index.close()

    def __enter__(self) -> "IngestionPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def ingest_file(self, path: str | Path, force: bool = False) -> IngestionResult:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise IngestionError(f"Input is not a file: {source}")
        parser = self.registry.parser_for(source)
        size = source.stat().st_size
        if size > self.config.max_file_bytes:
            raise FileTooLargeError(
                f"File is {size} bytes; limit is {self.config.max_file_bytes}: {source}"
            )

        document_id = stable_document_id(source)
        content_hash = sha256_file(source)
        signature = self.pipeline_signature
        if not force and self.index.is_current(document_id, content_hash, signature):
            existing = self.index.get_document(document_id)
            assert existing is not None
            return IngestionResult(
                document_id=document_id,
                source_path=str(source),
                stored_path=existing["stored_path"],
                content_hash=content_hash,
                status="skipped",
                block_count=existing["block_count"],
                chunk_count=existing["chunk_count"],
                warnings=json.loads(existing["warnings_json"]),
            )

        stored_path = self.artifacts.store_raw(document_id, content_hash, source)
        document = parser.parse(stored_path)
        document.source_path = source
        document.metadata.update(
            {"original_filename": source.name, "source_size_bytes": size}
        )
        document = self.cleaner.clean(document)
        if not document.blocks:
            raise EmptyDocumentError(f"No indexable text found in: {source}")
        chunks = self.chunker.chunk(document_id, document)
        if not chunks:
            raise EmptyDocumentError(f"No chunks generated from: {source}")
        vectors = self._embed_batches(chunk.text for chunk in chunks)
        if any(not any(vector) for vector in vectors):
            raise EmptyDocumentError(f"At least one chunk has no embeddable tokens: {source}")

        processed_path = self.artifacts.store_processed(
            document_id, document, chunks, content_hash, signature
        )
        self.index.replace_document(
            document_id=document_id,
            source_path=str(source),
            stored_path=str(stored_path.resolve()),
            processed_path=str(processed_path.resolve()),
            media_type=document.media_type,
            content_hash=content_hash,
            pipeline_signature=signature,
            embedding_model=self.embedder.name,
            embedding_dimensions=self.embedder.dimensions,
            metadata=document.metadata,
            warnings=document.warnings,
            block_count=len(document.blocks),
            chunks=chunks,
            vectors=vectors,
        )
        return IngestionResult(
            document_id=document_id,
            source_path=str(source),
            stored_path=str(stored_path.resolve()),
            content_hash=content_hash,
            status="indexed",
            block_count=len(document.blocks),
            chunk_count=len(chunks),
            warnings=document.warnings,
        )

    def discover(self, path: str | Path, recursive: bool = True) -> list[Path]:
        target = Path(path).expanduser().resolve()
        if target.is_file():
            return [target]
        if not target.is_dir():
            raise IngestionError(f"Input path does not exist: {target}")
        iterator = target.rglob("*") if recursive else target.glob("*")
        supported = set(self.registry.supported_extensions)
        return sorted(
            (item for item in iterator if item.is_file() and item.suffix.lower() in supported),
            key=lambda item: str(item).casefold(),
        )

    def search(self, query: str, limit: int = 5, document_id: str | None = None) -> list[SearchResult]:
        if not query.strip():
            return []
        vector = self.embedder.embed([query])[0]
        return self.index.search(
            vector,
            limit=limit,
            document_id=document_id,
            embedding_model=self.embedder.name,
        )

    def validate(self, check_files: bool = True) -> ValidationReport:
        return validate_index(self.index, check_files=check_files)

    def _embed_batches(self, texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
        result: list[list[float]] = []
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= batch_size:
                result.extend(self.embedder.embed(batch))
                batch.clear()
        if batch:
            result.extend(self.embedder.embed(batch))
        return result
