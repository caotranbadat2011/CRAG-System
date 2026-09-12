from __future__ import annotations

import json
from pathlib import Path

import pytest

from crag_ingestion.config import ChunkingConfig, IngestionConfig
from crag_ingestion.exceptions import EmptyDocumentError, FileTooLargeError, UnsupportedFormatError
from crag_ingestion.pipeline import IngestionPipeline


def _config(tmp_path: Path, **kwargs: object) -> IngestionConfig:
    return IngestionConfig(
        data_dir=tmp_path / "data",
        embedding_dimensions=128,
        chunking=ChunkingConfig(max_chars=220, overlap_chars=30, min_chars=30),
        **kwargs,
    )


def test_end_to_end_ingest_search_validate_and_skip(tmp_path: Path) -> None:
    source = tmp_path / "policy.md"
    source.write_text(
        "# Chính sách hoàn tiền\nKhách hàng được hoàn tiền trong vòng 30 ngày.\n\n"
        "# Bảo hành\nSản phẩm được bảo hành trong 12 tháng.",
        encoding="utf-8",
    )
    with IngestionPipeline(_config(tmp_path)) as pipeline:
        result = pipeline.ingest_file(source)
        assert result.status == "indexed"
        assert result.chunk_count >= 1
        assert Path(result.stored_path).read_bytes() == source.read_bytes()

        indexed = pipeline.index.get_document(result.document_id)
        assert indexed is not None
        processed = json.loads(Path(indexed["processed_path"]).read_text(encoding="utf-8"))
        assert processed["content_hash"] == result.content_hash
        assert all(chunk["document_id"] == result.document_id for chunk in processed["chunks"])

        hits = pipeline.search("hoàn tiền 30 ngày", limit=1)
        assert hits and hits[0].chunk.document_id == result.document_id
        assert "hoàn tiền" in hits[0].chunk.text

        report = pipeline.validate()
        assert report.valid, report.to_dict()
        assert report.document_count == 1
        assert report.chunk_count == result.chunk_count

        skipped = pipeline.ingest_file(source)
        assert skipped.status == "skipped"
        assert skipped.document_id == result.document_id


def test_changed_document_replaces_old_chunks(tmp_path: Path) -> None:
    source = tmp_path / "guide.txt"
    source.write_text("alpha " * 150, encoding="utf-8")
    with IngestionPipeline(_config(tmp_path)) as pipeline:
        first = pipeline.ingest_file(source)
        assert first.chunk_count > 1
        source.write_text("beta mới", encoding="utf-8")
        second = pipeline.ingest_file(source)
        assert second.document_id == first.document_id
        assert second.content_hash != first.content_hash
        assert second.chunk_count == 1
        texts = [row[0] for row in pipeline.index.connection.execute("SELECT text FROM chunks")]
        assert texts == ["beta mới"]


def test_file_size_empty_and_unsupported_guards(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_text("too large", encoding="utf-8")
    config = _config(tmp_path, max_file_bytes=2)
    with IngestionPipeline(config) as pipeline:
        with pytest.raises(FileTooLargeError):
            pipeline.ingest_file(large)

    empty = tmp_path / "empty.txt"
    empty.write_text(" \n\n ", encoding="utf-8")
    with IngestionPipeline(_config(tmp_path / "empty-case")) as pipeline:
        with pytest.raises(EmptyDocumentError):
            pipeline.ingest_file(empty)

        unsupported = tmp_path / "image.png"
        unsupported.write_bytes(b"png")
        with pytest.raises(UnsupportedFormatError):
            pipeline.ingest_file(unsupported)


def test_validation_detects_vector_corruption(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("nội dung hợp lệ", encoding="utf-8")
    with IngestionPipeline(_config(tmp_path)) as pipeline:
        pipeline.ingest_file(source)
        with pipeline.index.connection:
            pipeline.index.connection.execute("UPDATE chunks SET vector = ?", (b"\x00\x00\x00\x00",))
        report = pipeline.validate()
        assert not report.valid
        assert {issue.code for issue in report.issues} >= {"vector_dimensions", "vector_norm"}


def test_validation_detects_raw_and_processed_artifact_corruption(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("nội dung để kiểm tra checksum", encoding="utf-8")
    with IngestionPipeline(_config(tmp_path)) as pipeline:
        result = pipeline.ingest_file(source)
        document = pipeline.index.get_document(result.document_id)
        assert document is not None
        Path(document["stored_path"]).write_text("đã bị sửa", encoding="utf-8")
        processed_path = Path(document["processed_path"])
        processed = json.loads(processed_path.read_text(encoding="utf-8"))
        processed["content_hash"] = "wrong"
        processed_path.write_text(json.dumps(processed), encoding="utf-8")
        report = pipeline.validate()
        assert not report.valid
        assert {issue.code for issue in report.issues} >= {"raw_checksum", "processed_artifact"}
