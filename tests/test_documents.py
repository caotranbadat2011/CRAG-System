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


def test_failed_index_update_restores_source_and_original_index(tmp_path: Path, fake_embedder: object, monkeypatch: pytest.MonkeyPatch) -> None:
    with IngestionPipeline(IngestionConfig(data_dir=tmp_path / "data"), embedder=fake_embedder) as pipeline:
        manager = DocumentManager(pipeline)
        original_bytes = b"Old source about source arbitration."
        added = manager.upload("document.txt", original_bytes)
        document_id = added["document_id"]
        before = pipeline.index.document_chunks(document_id)
        delete = pipeline.index.client.delete
        failed = False

        def fail_once(*a: object, **kw: object) -> object:
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("transient delete failure")
            return delete(*a, **kw)

        monkeypatch.setattr(pipeline.index.client, "delete", fail_once)
        with pytest.raises(RuntimeError, match="transient delete failure"):
            manager.update_upload(document_id, "document.txt", b"New source about a different subject.")
        document = manager.get_document(document_id)
        assert Path(document["source_path"]).read_bytes() == original_bytes
        assert document["index_status"] == "current"
        restored = pipeline.index.document_chunks(document_id)
        assert len(restored) == len(before)
        for current, prior in zip(restored, before, strict=True):
            # Cosine storage may normalize floating-point vectors again.
            assert current.pop("vector") == pytest.approx(prior.pop("vector"))
            assert current == prior
        assert pipeline.validate().valid


def test_failed_first_upload_keeps_source_when_index_needs_recovery(tmp_path: Path, fake_embedder: object, monkeypatch: pytest.MonkeyPatch) -> None:
    with IngestionPipeline(IngestionConfig(data_dir=tmp_path / "data"), embedder=fake_embedder) as pipeline:
        manager = DocumentManager(pipeline)
        upsert = pipeline.index.client.upsert

        def fail_chunk_write(*a: object, **kw: object) -> object:
            if kw["points"][0].payload["record_type"] == "chunk":
                raise RuntimeError("write failed")
            return upsert(*a, **kw)

        def fail_cleanup(*a: object, **kw: object) -> object:
            raise RuntimeError("delete failed")

        with monkeypatch.context() as patch:
            patch.setattr(pipeline.index.client, "upsert", fail_chunk_write)
            patch.setattr(pipeline.index.client, "delete", fail_cleanup)
            with pytest.raises(RuntimeError, match="rollback is incomplete"):
                manager.upload("source.txt", b"Source retained for recovery.")
        pending = manager.list_documents()
        assert len(pending) == 1 and pending[0]["index_status"] == "incomplete_update"
        assert Path(pending[0]["source_path"]).read_bytes() == b"Source retained for recovery."
        manager.refresh(pending[0]["document_id"])
        assert manager.list_documents()[0]["index_status"] == "current"
        assert pipeline.validate().valid


def test_failed_source_restore_does_not_delete_backup(tmp_path: Path, fake_embedder: object, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    with IngestionPipeline(IngestionConfig(data_dir=tmp_path / "data"), embedder=fake_embedder) as pipeline:
        manager = DocumentManager(pipeline)
        added = manager.upload("source.txt", b"Original source remains recoverable.")
        source = Path(added["source_path"])
        backup = source.with_name(source.name + ".crag-backup")
        replace_file = os.replace

        def fail_restore(src: object, dst: object) -> None:
            if Path(src) == backup:
                raise OSError("source locked")
            replace_file(src, dst)

        def fail_ingest(*a: object, **kw: object) -> object:
            raise RuntimeError("ingestion failed")

        monkeypatch.setattr("crag_ingestion.documents.os.replace", fail_restore)
        monkeypatch.setattr(pipeline, "ingest_file", fail_ingest)
        with pytest.raises(OSError, match="source locked"):
            manager.update_upload(added["document_id"], "source.txt", b"Replacement.")
        assert backup.read_bytes() == b"Original source remains recoverable."
