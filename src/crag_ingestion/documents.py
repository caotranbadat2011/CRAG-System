from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .pipeline import IngestionPipeline
from .utils import sha256_file, stable_document_id


class DocumentManager:
    """Manage indexed documents and files uploaded through the local web app."""

    def __init__(self, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline
        self.upload_root = (pipeline.config.data_dir / "uploads").resolve()

    def list_documents(self) -> list[dict[str, Any]]:
        return [self._present(row) for row in self.pipeline.index.documents()]

    def get_document(self, document_id: str) -> dict[str, Any]:
        self._check_id(document_id)
        document = self.pipeline.index.get_document(document_id)
        if document is None:
            raise KeyError("Document was not found")
        return self._present(document)

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        name = self._check_upload(filename, content)
        directory = self.upload_root / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        source = directory / name
        try:
            self._write_atomic(source, content)
            result = self.pipeline.ingest_file(source)
            return result.to_dict()
        except Exception:
            shutil.rmtree(directory)
            raise

    def update_upload(self, document_id: str, filename: str, content: bytes) -> dict[str, Any]:
        document = self.get_document(document_id)
        name = self._check_upload(filename, content)
        source = self._managed_upload_path(document)
        if source is None or not source.is_file():
            raise ValueError("Only files uploaded through this web app can be replaced")
        if Path(name).suffix.lower() != source.suffix.lower():
            raise ValueError("Replacement file must have the same extension")
        backup = source.with_name(source.name + ".crag-backup")
        if backup.exists():
            raise RuntimeError("An unfinished document update needs manual recovery")
        shutil.copy2(source, backup)
        try:
            self._write_atomic(source, content)
            result = self.pipeline.ingest_file(source, force=True)
            if result.document_id != document_id:
                raise RuntimeError("Document ID changed during replacement")
            return result.to_dict()
        except Exception:
            os.replace(backup, source)
            raise
        finally:
            backup.unlink(missing_ok=True)

    def refresh(self, document_id: str) -> dict[str, Any]:
        document = self.get_document(document_id)
        source = Path(str(document["source_path"]))
        if not source.is_file() or source.is_symlink():
            raise ValueError("Original source file is missing or unsafe")
        if stable_document_id(source) != document_id:
            raise ValueError("Original source path no longer matches the document ID")
        return self.pipeline.ingest_file(source, force=True).to_dict()

    def delete(self, document_id: str) -> dict[str, Any]:
        document = self.get_document(document_id)
        managed_source = self._managed_upload_path(document)
        removed_chunks = self.pipeline.index.delete_document(document_id)
        self.pipeline.artifacts.delete_document(document_id)
        if managed_source is not None and managed_source.is_file():
            managed_source.unlink()
            try:
                managed_source.parent.rmdir()
            except OSError:
                # Keep other files (for example, an interrupted update backup)
                # for explicit recovery rather than removing an unknown target.
                pass
        return {"document_id": document_id, "removed_chunks": removed_chunks}

    def source_file(self, document_id: str) -> tuple[Path, str]:
        document = self.get_document(document_id)
        raw_root = self.pipeline.config.raw_dir.resolve()
        path = Path(str(document["stored_path"]))
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Stored source copy is missing")
        resolved = path.resolve()
        if not resolved.is_relative_to(raw_root) or resolved.parent.name != document_id:
            raise ValueError("Stored source copy is outside the managed raw directory")
        if sha256_file(resolved) != document["content_hash"]:
            raise ValueError("Stored source copy has an invalid content hash")
        return resolved, str(document["media_type"])

    def _check_upload(self, filename: str, content: bytes) -> str:
        if (
            not isinstance(filename, str) or not filename or len(filename) > 180
            or filename in {".", ".."} or Path(filename).name != filename
            or any(char in filename for char in '/\\:*?"<>|')
            or any(ord(char) < 32 for char in filename)
            or filename.endswith((" ", "."))
        ):
            raise ValueError("Invalid upload filename")
        if Path(filename).suffix.lower() not in self.pipeline.registry.supported_extensions:
            raise ValueError("Unsupported document extension")
        if not content or len(content) > self.pipeline.config.max_file_bytes:
            raise ValueError("Upload is empty or exceeds the configured size limit")
        return filename

    @staticmethod
    def _check_id(document_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{24}", document_id):
            raise ValueError("Invalid document ID")

    def _managed_upload_path(self, document: dict[str, Any]) -> Path | None:
        source = Path(str(document["source_path"]))
        root = self.upload_root.resolve()
        if source.is_symlink() or not source.is_relative_to(self.upload_root):
            return None
        resolved = source.resolve()
        if resolved.parent.parent != root or not re.fullmatch(r"[0-9a-f]{32}", resolved.parent.name):
            return None
        if stable_document_id(resolved) != document["document_id"]:
            return None
        return resolved

    def _present(self, document: dict[str, Any]) -> dict[str, Any]:
        return {
            **document,
            "managed_upload": self._managed_upload_path(document) is not None,
            "index_status": self.pipeline.document_index_status(document),
        }

    @staticmethod
    def _write_atomic(destination: Path, content: bytes) -> None:
        fd, name = tempfile.mkstemp(prefix="upload-", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
