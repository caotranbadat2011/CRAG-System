from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .models import Chunk, ParsedDocument
from .utils import sha256_file


class ArtifactStore:
    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def store_raw(self, document_id: str, content_hash: str, source: Path) -> Path:
        directory = self.raw_dir / document_id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{content_hash}{source.suffix.lower()}"
        if destination.is_file() and sha256_file(destination) == content_hash:
            return destination
        fd, temporary_name = tempfile.mkstemp(prefix="upload-", dir=directory)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def store_processed(
        self,
        document_id: str,
        document: ParsedDocument,
        chunks: list[Chunk],
        content_hash: str,
        pipeline_signature: str,
    ) -> Path:
        directory = self.processed_dir / document_id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{content_hash}.json"
        payload = {
            "schema_version": 1,
            "document_id": document_id,
            "source_path": str(document.source_path),
            "media_type": document.media_type,
            "content_hash": content_hash,
            "pipeline_signature": pipeline_signature,
            "metadata": document.metadata,
            "warnings": document.warnings,
            "blocks": [block.to_dict() for block in document.blocks],
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix="processed-", suffix=".json", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
