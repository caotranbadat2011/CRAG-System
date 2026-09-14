from __future__ import annotations

from pathlib import Path

import pytest

from crag_ingestion.config import IngestionConfig
from crag_ingestion.documents import DocumentManager
from crag_ingestion.exceptions import EmptyDocumentError
from crag_ingestion.pipeline import IngestionPipeline


def test_managed_upload_update_refresh_and_delete(tmp_path: Path, fake_embedder: object) -> None:
    config = IngestionConfig(data_dir=tmp_path / "data")
    with IngestionPipeline(config, embedder=fake_embedder) as pipeline:  # type: ignore[arg-type]
        manager = DocumentManager(pipeline)
        added = manager.upload("huong-dan.md", b"# Huong dan\nChinh sach hoan tien trong 30 ngay.")
        doc_id = added["document_id"]
        original = manager.get_document(doc_id)
        source = Path(original["source_path"])
        assert original["managed_upload"] is True
        assert manager.source_file(doc_id)[0].read_bytes() == source.read_bytes()
        assert pipeline.index.count_document_chunks(doc_id) == added["chunk_count"]

        updated = manager.update_upload(doc_id, "replacement.md", b"# Moi\nBao hanh trong 12 thang.")
        assert updated["document_id"] == doc_id
        assert updated["content_hash"] != added["content_hash"]
        assert "Bao hanh" in pipeline.index.document_chunks(doc_id)[0]["text"]
        assert manager.source_file(doc_id)[0].read_bytes() == source.read_bytes()

        with pytest.raises(ValueError):
            manager.update_upload(doc_id, "wrong.txt", b"different type")
        with pytest.raises(EmptyDocumentError):
            manager.update_upload(doc_id, "replacement.md", b" \n ")
        assert source.read_bytes() == b"# Moi\nBao hanh trong 12 thang."
        assert "Bao hanh" in pipeline.index.document_chunks(doc_id)[0]["text"]

        source.write_text("# Moi\nBao hanh trong 24 thang.", encoding="utf-8")
        refreshed = manager.refresh(doc_id)
        assert refreshed["content_hash"] != updated["content_hash"]
        assert "24 thang" in pipeline.index.document_chunks(doc_id)[0]["text"]

        raw_dir = config.raw_dir / doc_id
        processed_dir = config.processed_dir / doc_id
        deleted = manager.delete(doc_id)
        assert deleted["removed_chunks"] == refreshed["chunk_count"]
        assert pipeline.index.get_document(doc_id) is None
        assert not source.exists() and not raw_dir.exists() and not processed_dir.exists()


def test_external_source_is_preserved_when_removed_from_index(
    tmp_path: Path, fake_embedder: object,
) -> None:
    source = tmp_path / "user-owned.txt"
    source.write_text("Noi dung tai lieu goc.", encoding="utf-8")
    with IngestionPipeline(IngestionConfig(data_dir=tmp_path / "data"), embedder=fake_embedder) as pipeline:  # type: ignore[arg-type]
        indexed = pipeline.ingest_file(source)
        manager = DocumentManager(pipeline)
        assert manager.get_document(indexed.document_id)["managed_upload"] is False
        with pytest.raises(ValueError, match="uploaded through this web app"):
            manager.update_upload(indexed.document_id, "user-owned.txt", b"changed")
        manager.delete(indexed.document_id)
        assert source.read_text(encoding="utf-8") == "Noi dung tai lieu goc."
        assert pipeline.index.get_document(indexed.document_id) is None


def test_document_manager_rejects_unsafe_uploads_and_ids(
    tmp_path: Path, fake_embedder: object,
) -> None:
    with IngestionPipeline(IngestionConfig(data_dir=tmp_path / "data"), embedder=fake_embedder) as pipeline:  # type: ignore[arg-type]
        manager = DocumentManager(pipeline)
        for name in ("../escape.txt", "bad/name.md", "bad.exe", "bad?.txt"):
            with pytest.raises(ValueError):
                manager.upload(name, b"content")
        with pytest.raises(ValueError, match="Invalid document ID"):
            manager.delete("../other")
        with pytest.raises(KeyError, match="not found"):
            manager.get_document("a" * 24)
