from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import models

from crag_ingestion.embeddings.base import EmbeddingVector
from crag_ingestion.index import QdrantVectorIndex
from crag_ingestion.models import Chunk


def _chunk(chunk_id: str, document_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        ordinal=ordinal,
        text=text,
        char_count=len(text),
        pages=(1,),
        heading_path=("Heading",),
        metadata={"source": {"page": 1}},
    )


def _replace(
    index: QdrantVectorIndex,
    *,
    document_id: str,
    source_path: str,
    chunk: Chunk,
    vector: EmbeddingVector,
) -> None:
    index.replace_document(
        document_id=document_id,
        source_path=source_path,
        stored_path=f"{source_path}.raw",
        processed_path=f"{source_path}.json",
        media_type="text/plain",
        content_hash="content-hash",
        pipeline_signature="pipeline-signature",
        embedding_model="test/model",
        embedding_dimensions=len(vector.dense),
        metadata={"title": "Test"},
        warnings=[],
        block_count=1,
        chunks=[chunk],
        vectors=[vector],
    )


def test_qdrant_upsert_filter_and_document_replacement(tmp_path: Path) -> None:
    with QdrantVectorIndex(path=tmp_path / "qdrant") as index:
        assert index.collection_name == "crag_bge_m3_hybrid"
        first = _chunk("1" * 32, "doc-1", 0, "alpha")
        second = _chunk("2" * 32, "doc-2", 0, "beta")
        _replace(
            index,
            document_id="doc-1",
            source_path="/documents/one.txt",
            chunk=first,
            vector=EmbeddingVector([1.0, 0.0, 0.0], {11: 0.7}),
        )
        _replace(
            index,
            document_id="doc-2",
            source_path="/documents/two.txt",
            chunk=second,
            vector=EmbeddingVector([0.0, 1.0, 0.0], {22: 1.5}),
        )

        assert index.collection_schema() == {
            "exists": True,
            "status": "green",
            "vector_size": 3,
            "distance": "Cosine",
            "points_count": 2,
            "hybrid_vectors": True,
        }
        assert {document["document_id"] for document in index.documents()} == {
            "doc-1",
            "doc-2",
        }
        filtered = index.search(
            EmbeddingVector([0.0, 1.0, 0.0], {22: 1.0}), document_id="doc-2"
        )
        assert len(filtered) == 1
        assert filtered[0].chunk.document_id == "doc-2"
        assert index.search(
            EmbeddingVector([0.0, 1.0, 0.0], {22: 1.0}),
            document_id="doc-2", pipeline_signature="outdated-signature",
        ) == []
        assert len(index.search(
            EmbeddingVector([0.0, 1.0, 0.0], {22: 1.0}),
            document_id="doc-2", pipeline_signature="pipeline-signature",
        )) == 1

        replacement = _chunk("3" * 32, "doc-1", 0, "alpha updated")
        _replace(
            index,
            document_id="doc-1",
            source_path="/documents/one.txt",
            chunk=replacement,
            vector=EmbeddingVector([1.0, 0.0, 0.0], {11: 0.9}),
        )
        assert [row["text"] for row in index.document_chunks("doc-1")] == [
            "alpha updated"
        ]
        assert index.collection_schema()["points_count"] == 2
        row = index.document_chunks("doc-1")[0]
        assert row["sparse_vector"] == {"indices": [11], "values": [0.9]}
        assert index.delete_document("doc-1") == 1
        assert index.get_document("doc-1") is None
        assert index.get_document("doc-2") is not None
        assert index.delete_document("doc-1") == 0


def test_qdrant_rejects_an_incompatible_embedding_dimension(tmp_path: Path) -> None:
    with QdrantVectorIndex(path=tmp_path / "qdrant") as index:
        index.ensure_collection(384)
        with pytest.raises(ValueError, match="expects 384 dimensions"):
            index.ensure_collection(768)


def test_qdrant_rejects_missing_lexical_weights_before_creating_collection(
    tmp_path: Path,
) -> None:
    with QdrantVectorIndex(path=tmp_path / "qdrant") as index:
        with pytest.raises(ValueError, match="lexical_weights"):
            _replace(
                index, document_id="bad-doc", source_path="/bad.txt",
                chunk=_chunk("6" * 32, "bad-doc", 0, "bad"),
                vector=EmbeddingVector([1.0, 0.0, 0.0], {}),
            )
        assert not index.collection_exists()


def test_qdrant_rejects_dense_only_collection_without_overwriting(tmp_path: Path) -> None:
    with QdrantVectorIndex(path=tmp_path / "qdrant") as index:
        index.client.create_collection(
            collection_name=index.collection_name,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        with pytest.raises(ValueError, match="incompatible vector schema"):
            index.ensure_collection(3)
        assert index.collection_schema()["hybrid_vectors"] is False


def test_hybrid_search_fuses_dense_and_lexical_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with QdrantVectorIndex(path=tmp_path / "qdrant") as index:
        _replace(
            index, document_id="dense-doc", source_path="/dense.txt",
            chunk=_chunk("4" * 32, "dense-doc", 0, "dense only"),
            vector=EmbeddingVector([1.0, 0.0, 0.0], {10: 1.0}),
        )
        _replace(
            index, document_id="sparse-doc", source_path="/sparse.txt",
            chunk=_chunk("5" * 32, "sparse-doc", 0, "lexical match"),
            vector=EmbeddingVector([0.0, 1.0, 0.0], {99: 4.0}),
        )
        query_calls: list[dict[str, object]] = []
        original_query = index.client.query_points

        def captured_query(*args: object, **kwargs: object) -> object:
            query_calls.append(kwargs)
            return original_query(*args, **kwargs)

        monkeypatch.setattr(index.client, "query_points", captured_query)
        hits = index.search(EmbeddingVector([1.0, 0.0, 0.0], {99: 1.0}), limit=2)
        assert {hit.chunk.document_id for hit in hits} == {"dense-doc", "sparse-doc"}
        assert isinstance(query_calls[0]["query"], models.FusionQuery)
        assert query_calls[0]["query"].fusion == models.Fusion.RRF
        assert [prefetch.using for prefetch in query_calls[0]["prefetch"]] == [
            "dense", "sparse"
        ]
        filtered = index.search(
            EmbeddingVector([1.0, 0.0, 0.0], {99: 1.0}),
            document_id="sparse-doc", limit=2,
        )
        assert [hit.chunk.document_id for hit in filtered] == ["sparse-doc"]
