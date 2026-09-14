from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from qdrant_client import QdrantClient, models

from ..config import DEFAULT_QDRANT_COLLECTION
from ..embeddings.base import EmbeddingVector
from ..models import Chunk, SearchResult


class QdrantVectorIndex:
    """Qdrant-backed chunk index with document metadata stored in point payloads."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        timeout: int = 30,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.url = url
        self.path = path
        self._owns_client = client is None
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        else:
            if path is None:
                raise ValueError("A local Qdrant path is required when qdrant_url is not set")
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "QdrantVectorIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("Qdrant vector size must be positive")
        if self.collection_exists():
            schema = self.collection_schema()
            if not schema["hybrid_vectors"]:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' has an incompatible vector schema; "
                    "use a new hybrid collection and re-ingest documents"
                )
            if schema["vector_size"] != vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' expects "
                    f"{schema['vector_size']} dimensions, but the embedding model returns "
                    f"{vector_size}; use a different collection or rebuild it"
                )
            if schema["distance"] != models.Distance.COSINE.value:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' must use Cosine distance"
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        if self.url:
            for field_name in (
                "record_type", "document_id", "source_path", "embedding_model",
                "pipeline_signature",
            ):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    def collection_schema(self) -> dict[str, object]:
        if not self.collection_exists():
            return {
                "exists": False,
                "status": None,
                "vector_size": None,
                "distance": None,
                "points_count": 0,
                "hybrid_vectors": False,
            }
        info = self.client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        sparse_config = info.config.params.sparse_vectors or {}
        dense_config = vector_config.get("dense") if isinstance(vector_config, dict) else vector_config
        distance = (
            getattr(dense_config.distance, "value", str(dense_config.distance))
            if dense_config is not None else None
        )
        status = getattr(info.status, "value", str(info.status))
        count = self.client.count(self.collection_name, exact=True).count
        return {
            "exists": True,
            "status": status,
            "vector_size": int(dense_config.size) if dense_config is not None else None,
            "distance": distance,
            "points_count": int(count),
            "hybrid_vectors": (
                isinstance(vector_config, dict)
                and set(vector_config) == {"dense"}
                and set(sparse_config) == {"sparse"}
            ),
        }

    def is_current(self, document_id: str, content_hash: str, pipeline_signature: str) -> bool:
        document = self.get_document(document_id)
        return bool(
            document
            and document["content_hash"] == content_hash
            and document["pipeline_signature"] == pipeline_signature
        )

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        if not self.collection_exists():
            return None
        records, _ = self._scroll(
            self._filter(document_id=document_id), limit=1, with_vectors=False
        )
        if not records:
            return None
        return self._document_from_payload(records[0].payload or {})

    def replace_document(
        self,
        *,
        document_id: str,
        source_path: str,
        stored_path: str,
        processed_path: str,
        media_type: str,
        content_hash: str,
        pipeline_signature: str,
        embedding_model: str,
        embedding_dimensions: int,
        metadata: dict[str, Any],
        warnings: list[str],
        block_count: int,
        chunks: list[Chunk],
        vectors: list[EmbeddingVector],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk must have exactly one vector")
        if any(len(vector.dense) != embedding_dimensions for vector in vectors):
            raise ValueError("Embedding dimension mismatch")
        if any(
            not any(vector.dense) or not all(math.isfinite(value) for value in vector.dense)
            for vector in vectors
        ):
            raise ValueError("Every chunk must have a finite, nonzero dense vector")
        sparse_vectors = [self._sparse_vector(vector) for vector in vectors]
        self.ensure_collection(embedding_dimensions)

        existing_ids = self._point_ids(self._filter(source_path=source_path))
        existing_ids.update(self._point_ids(self._filter(document_id=document_id)))
        ingested_at = datetime.now(timezone.utc).isoformat()
        document_payload: dict[str, Any] = {
            "record_type": "chunk",
            "document_id": document_id,
            "source_path": source_path,
            "stored_path": stored_path,
            "processed_path": processed_path,
            "media_type": media_type,
            "content_hash": content_hash,
            "pipeline_signature": pipeline_signature,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "document_metadata": metadata,
            "warnings": warnings,
            "block_count": block_count,
            "chunk_count": len(chunks),
            "ingested_at": ingested_at,
        }
        points = [
            models.PointStruct(
                id=UUID(hex=chunk.chunk_id),
                vector={"dense": vector.dense, "sparse": sparse},
                payload={
                    **document_payload,
                    "chunk_id": chunk.chunk_id,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "pages": list(chunk.pages),
                    "heading_path": list(chunk.heading_path),
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector, sparse in zip(chunks, vectors, sparse_vectors, strict=True)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        new_ids = {point.id for point in points}
        stale_ids = list(existing_ids - new_ids)
        if stale_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=stale_ids),
                wait=True,
            )

    def search(
        self,
        query_vector: EmbeddingVector,
        limit: int = 5,
        document_id: str | None = None,
        embedding_model: str | None = None,
        pipeline_signature: str | None = None,
    ) -> list[SearchResult]:
        if (limit <= 0 or not any(query_vector.dense) or not query_vector.lexical_weights
                or not self.collection_exists()):
            return []
        schema = self.collection_schema()
        if not schema["hybrid_vectors"]:
            raise ValueError("Qdrant collection does not have dense and sparse named vectors")
        if schema["vector_size"] != len(query_vector.dense):
            raise ValueError(
                f"Query vector has {len(query_vector.dense)} dimensions; Qdrant collection "
                f"expects {schema['vector_size']}"
            )
        query_filter = self._filter(
            document_id=document_id, embedding_model=embedding_model,
            pipeline_signature=pipeline_signature,
        )
        prefetch_limit = max(limit * 4, 20)
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector.dense,
                    using="dense",
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=self._sparse_vector(query_vector),
                    using="sparse",
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = self._chunk_from_payload(payload)
            results.append(
                SearchResult(
                    score=float(point.score),
                    chunk=chunk,
                    source_path=str(payload["source_path"]),
                    stored_path=str(payload["stored_path"]),
                )
            )
        return results

    def documents(self) -> list[dict[str, Any]]:
        if not self.collection_exists():
            return []
        documents: dict[str, dict[str, Any]] = {}
        for record in self._scroll_all(self._filter(), with_vectors=False):
            payload = record.payload or {}
            document_id = payload.get("document_id")
            if isinstance(document_id, str) and document_id not in documents:
                documents[document_id] = self._document_from_payload(payload)
        return sorted(
            documents.values(),
            key=lambda value: (str(value["ingested_at"]), str(value["document_id"])),
        )

    def document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        if not self.collection_exists():
            return []
        rows = [
            self._diagnostic_row(record)
            for record in self._scroll_all(
                self._filter(document_id=document_id), with_vectors=True
            )
        ]
        return sorted(rows, key=lambda row: int(row["ordinal"]))

    def count_document_chunks(self, document_id: str) -> int:
        if not self.collection_exists():
            return 0
        return int(
            self.client.count(
                collection_name=self.collection_name,
                count_filter=self._filter(document_id=document_id),
                exact=True,
            ).count
        )

    def delete_document(self, document_id: str) -> int:
        """Remove only chunk points belonging to one existing document."""
        if not document_id or not self.collection_exists():
            return 0
        count = self.count_document_chunks(document_id)
        if count:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=self._filter(document_id=document_id)
                ),
                wait=True,
            )
        return count

    def diagnostic_rows(self) -> Iterable[dict[str, Any]]:
        if not self.collection_exists():
            return []
        return [
            self._diagnostic_row(record)
            for record in self._scroll_all(self._filter(), with_vectors=True)
        ]

    def _point_ids(self, query_filter: models.Filter) -> set[int | str | UUID]:
        return {
            record.id
            for record in self._scroll_all(query_filter, with_vectors=False)
        }

    def _scroll(
        self,
        query_filter: models.Filter,
        *,
        limit: int,
        with_vectors: bool,
        offset: int | str | UUID | None = None,
    ) -> tuple[list[Any], int | str | UUID | None]:
        return self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )

    def _scroll_all(
        self, query_filter: models.Filter, *, with_vectors: bool
    ) -> Iterable[Any]:
        offset: int | str | UUID | None = None
        while True:
            records, offset = self._scroll(
                query_filter,
                limit=256,
                offset=offset,
                with_vectors=with_vectors,
            )
            yield from records
            if offset is None:
                break

    @staticmethod
    def _filter(
        *,
        document_id: str | None = None,
        source_path: str | None = None,
        embedding_model: str | None = None,
        pipeline_signature: str | None = None,
    ) -> models.Filter:
        conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="record_type", match=models.MatchValue(value="chunk")
            )
        ]
        for key, value in (
            ("document_id", document_id),
            ("source_path", source_path),
            ("embedding_model", embedding_model),
            ("pipeline_signature", pipeline_signature),
        ):
            if value is not None:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions)

    @staticmethod
    def _document_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "document_id",
            "source_path",
            "stored_path",
            "processed_path",
            "media_type",
            "content_hash",
            "pipeline_signature",
            "embedding_model",
            "embedding_dimensions",
            "block_count",
            "chunk_count",
            "ingested_at",
        )
        return {
            **{key: payload[key] for key in keys},
            "metadata": payload.get("document_metadata", {}),
            "warnings": payload.get("warnings", []),
        }

    @staticmethod
    def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            ordinal=int(payload["ordinal"]),
            text=str(payload["text"]),
            char_count=int(payload["char_count"]),
            pages=tuple(int(page) for page in payload.get("pages", [])),
            heading_path=tuple(str(value) for value in payload.get("heading_path", [])),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def _diagnostic_row(cls, record: Any) -> dict[str, Any]:
        payload = dict(record.payload or {})
        vector = record.vector or {}
        if not isinstance(vector, dict):
            payload["vector"] = list(vector)
            payload["sparse_vector"] = None
            payload["point_id"] = record.id
            return payload
        payload["vector"] = list(vector.get("dense") or [])
        sparse = vector.get("sparse")
        payload["sparse_vector"] = {
            "indices": list(sparse.indices),
            "values": list(sparse.values),
        } if isinstance(sparse, models.SparseVector) else None
        payload["point_id"] = record.id
        return payload

    @staticmethod
    def _sparse_vector(vector: EmbeddingVector) -> models.SparseVector:
        pairs = sorted(vector.lexical_weights.items())
        if not pairs or any(
            not isinstance(token_id, int) or token_id < 0
            or not math.isfinite(weight) or weight <= 0
            for token_id, weight in pairs
        ):
            raise ValueError("Sparse vector requires nonempty, positive lexical_weights")
        return models.SparseVector(
            indices=[token_id for token_id, _ in pairs],
            values=[float(weight) for _, weight in pairs],
        )
