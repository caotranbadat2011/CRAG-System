from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_document_id(path: Path) -> str:
    canonical = os.path.normcase(str(path.resolve())).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:24]


def stable_chunk_id(document_id: str, ordinal: int, text: str) -> str:
    value = f"{document_id}\0{ordinal}\0{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:32]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

