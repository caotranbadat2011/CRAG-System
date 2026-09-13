from __future__ import annotations

from pathlib import Path

import pytest

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
    vector: list[float],
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
        embedding_dimensions=len(vector),
        metadata={"title": "Test"},
        warnings=[],
        block_count=1,
        chunks=[chunk],
        vectors=[vector],
    )


def test_qdrant_upsert_filter_and_document_replacement(tmp_path: Path) -> None:
    with QdrantVectorIndex(path=tmp_path / "qdrant") as index:
        assert index.collection_name == "crag_bge_m3"
        first = _chunk("1" * 32, "doc-1", 0, "alpha")
        second = _chunk("2" * 32, "doc-2", 0, "beta")
        _replace(
            index,
            document_id="doc-1",
            source_path="/documents/one.txt",
            chunk=first,
            vector=[1.0, 0.0, 0.0],
        )
        _replace(
            index,
            document_id="doc-2",
            source_path="/documents/two.txt",
            chunk=second,
            vector=[0.0, 1.0, 0.0],
        )

        assert index.collection_schema() == {
            "exists": True,
            "status": "green",
            "vector_size": 3,
            "distance": "Cosine",
            "points_count": 2,
        }
        assert {document["document_id"] for document in index.documents()} == {
            "doc-1",
            "doc-2",
        }
        filtered = index.search([0.0, 1.0, 0.0], document_id="doc-2")
        assert len(filtered) == 1
        assert filtered[0].chunk.document_id == "doc-2"

        replacement = _chunk("3" * 32, "doc-1", 0, "alpha updated")
        _replace(
            index,
            document_id="doc-1",
            source_path="/documents/one.txt",
            chunk=replacement,
            vector=[1.0, 0.0, 0.0],
        )
        assert [row["text"] for row in index.document_chunks("doc-1")] == [
            "alpha updated"
        ]
        assert index.collection_schema()["points_count"] == 2


def test_qdrant_rejects_an_incompatible_embedding_dimension(tmp_path: Path) -> None:
    with QdrantVectorIndex(path=tmp_path / "qdrant") as index:
        index.ensure_collection(384)
        with pytest.raises(ValueError, match="expects 384 dimensions"):
            index.ensure_collection(768)
