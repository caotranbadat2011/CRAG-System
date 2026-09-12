from __future__ import annotations

import json
import math
import sqlite3
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..models import Chunk, SearchResult

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    stored_path TEXT NOT NULL,
    processed_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    pipeline_signature TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL CHECK (embedding_dimensions > 0),
    metadata_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    block_count INTEGER NOT NULL CHECK (block_count >= 0),
    chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    char_count INTEGER NOT NULL CHECK (char_count > 0),
    pages_json TEXT NOT NULL,
    heading_path_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    vector BLOB NOT NULL,
    UNIQUE(document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


class SQLiteVectorIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(_SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteVectorIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def encode_vector(vector: list[float]) -> bytes:
        values = array("f", vector)
        if values.itemsize != 4:
            raise RuntimeError("Unexpected float storage size")
        return values.tobytes()

    @staticmethod
    def decode_vector(blob: bytes) -> list[float]:
        values = array("f")
        values.frombytes(blob)
        return values.tolist()

    def is_current(self, document_id: str, content_hash: str, pipeline_signature: str) -> bool:
        row = self.connection.execute(
            "SELECT content_hash, pipeline_signature FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return bool(
            row
            and row["content_hash"] == content_hash
            and row["pipeline_signature"] == pipeline_signature
        )

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()

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
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk must have exactly one vector")
        if any(len(vector) != embedding_dimensions for vector in vectors):
            raise ValueError("Embedding dimension mismatch")

        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            # Handle a source path previously associated with a legacy document id.
            self.connection.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))
            self.connection.execute(
                """INSERT INTO documents (
                    document_id, source_path, stored_path, processed_path, media_type,
                    content_hash, pipeline_signature, embedding_model, embedding_dimensions,
                    metadata_json, warnings_json, block_count, chunk_count, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    source_path,
                    stored_path,
                    processed_path,
                    media_type,
                    content_hash,
                    pipeline_signature,
                    embedding_model,
                    embedding_dimensions,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    json.dumps(warnings, ensure_ascii=False),
                    block_count,
                    len(chunks),
                    now,
                ),
            )
            self.connection.executemany(
                """INSERT INTO chunks (
                    chunk_id, document_id, ordinal, text, char_count, pages_json,
                    heading_path_json, metadata_json, vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.chunk_id,
                        document_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.char_count,
                        json.dumps(chunk.pages),
                        json.dumps(chunk.heading_path, ensure_ascii=False),
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                        self.encode_vector(vector),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_id: str | None = None,
        embedding_model: str | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if not query_norm:
            return []
        sql = """SELECT c.*, d.source_path, d.stored_path, d.embedding_dimensions
                 FROM chunks c JOIN documents d USING(document_id)"""
        filters: list[str] = []
        parameter_values: list[Any] = []
        if document_id:
            filters.append("c.document_id = ?")
            parameter_values.append(document_id)
        if embedding_model:
            filters.append("d.embedding_model = ?")
            parameter_values.append(embedding_model)
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        scored: list[SearchResult] = []
        for row in self.connection.execute(sql, tuple(parameter_values)):
            if row["embedding_dimensions"] != len(query_vector):
                continue
            vector = self.decode_vector(row["vector"])
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if not vector_norm:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector, strict=True)) / (
                query_norm * vector_norm
            )
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                ordinal=row["ordinal"],
                text=row["text"],
                char_count=row["char_count"],
                pages=tuple(json.loads(row["pages_json"])),
                heading_path=tuple(json.loads(row["heading_path_json"])),
                metadata=json.loads(row["metadata_json"]),
            )
            scored.append(SearchResult(score, chunk, row["source_path"], row["stored_path"]))
        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit]

    def documents(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT document_id, source_path, stored_path, processed_path, media_type,
                      content_hash, pipeline_signature, embedding_model, embedding_dimensions,
                      block_count, chunk_count, warnings_json, ingested_at
               FROM documents ORDER BY ingested_at, document_id"""
        ).fetchall()
        return [
            {**dict(row), "warnings": json.loads(row["warnings_json"])}
            for row in rows
        ]

    def diagnostic_rows(self) -> Iterable[sqlite3.Row]:
        return self.connection.execute(
            """SELECT c.*, d.embedding_dimensions, d.source_path, d.processed_path,
                      d.stored_path, d.media_type
               FROM chunks c JOIN documents d USING(document_id)
               ORDER BY c.document_id, c.ordinal"""
        )
