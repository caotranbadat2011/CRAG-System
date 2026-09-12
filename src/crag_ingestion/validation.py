from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .index import SQLiteVectorIndex
from .utils import sha256_file


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    document_id: str | None = None
    chunk_id: str | None = None


@dataclass(slots=True)
class ValidationReport:
    valid: bool
    document_count: int
    chunk_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_index(index: SQLiteVectorIndex, check_files: bool = True) -> ValidationReport:
    issues: list[ValidationIssue] = []
    documents = index.documents()
    chunk_count = 0
    seen_ordinals: dict[str, list[int]] = {}

    integrity = index.connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        issues.append(ValidationIssue("error", "sqlite_integrity", str(integrity)))
    for row in index.connection.execute("PRAGMA foreign_key_check"):
        issues.append(ValidationIssue("error", "foreign_key", repr(tuple(row))))

    for document in documents:
        document_id = document["document_id"]
        if check_files:
            for field_name in ("stored_path", "processed_path"):
                if not Path(document[field_name]).is_file():
                    issues.append(
                        ValidationIssue(
                            "error", "missing_artifact", f"Missing {field_name}: {document[field_name]}", document_id
                        )
                    )
            stored_path = Path(document["stored_path"])
            if stored_path.is_file() and sha256_file(stored_path) != document["content_hash"]:
                issues.append(
                    ValidationIssue(
                        "error", "raw_checksum", "Stored source checksum does not match the index", document_id
                    )
                )
            processed_path = Path(document["processed_path"])
            if processed_path.is_file():
                try:
                    processed = json.loads(processed_path.read_text(encoding="utf-8"))
                    expected = {
                        "document_id": document_id,
                        "content_hash": document["content_hash"],
                        "pipeline_signature": document["pipeline_signature"],
                    }
                    for key, expected_value in expected.items():
                        if processed.get(key) != expected_value:
                            raise ValueError(f"processed {key} does not match the index")
                    if len(processed.get("chunks", [])) != document["chunk_count"]:
                        raise ValueError("processed chunk count does not match the index")
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    issues.append(
                        ValidationIssue("error", "processed_artifact", str(exc), document_id)
                    )
        actual = index.connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchone()[0]
        if actual != document["chunk_count"]:
            issues.append(
                ValidationIssue(
                    "error",
                    "chunk_count_mismatch",
                    f"metadata={document['chunk_count']}, actual={actual}",
                    document_id,
                )
            )

    for row in index.diagnostic_rows():
        chunk_count += 1
        document_id = row["document_id"]
        chunk_id = row["chunk_id"]
        seen_ordinals.setdefault(document_id, []).append(row["ordinal"])
        vector = index.decode_vector(row["vector"])
        if len(vector) != row["embedding_dimensions"]:
            issues.append(
                ValidationIssue("error", "vector_dimensions", f"expected {row['embedding_dimensions']}, got {len(vector)}", document_id, chunk_id)
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or not 0.98 <= norm <= 1.02:
            issues.append(
                ValidationIssue("error", "vector_norm", f"L2 norm is {norm:.6f}", document_id, chunk_id)
            )
        if row["char_count"] != len(row["text"]):
            issues.append(
                ValidationIssue("error", "char_count", "Stored char count is incorrect", document_id, chunk_id)
            )
        try:
            pages = json.loads(row["pages_json"])
            headings = json.loads(row["heading_path_json"])
            metadata = json.loads(row["metadata_json"])
            if not isinstance(pages, list) or not all(isinstance(page, int) and page > 0 for page in pages):
                raise ValueError("pages must be positive integers")
            if not isinstance(headings, list) or not all(isinstance(value, str) for value in headings):
                raise ValueError("heading path must contain strings")
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            issues.append(ValidationIssue("error", "invalid_provenance", str(exc), document_id, chunk_id))
        if row["media_type"] == "application/pdf" and not json.loads(row["pages_json"]):
            issues.append(
                ValidationIssue("warning", "pdf_page_missing", "PDF chunk has no page reference", document_id, chunk_id)
            )

    for document_id, ordinals in seen_ordinals.items():
        if ordinals != list(range(len(ordinals))):
            issues.append(
                ValidationIssue("error", "ordinal_gap", f"ordinals={ordinals}", document_id)
            )
    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        document_count=len(documents),
        chunk_count=chunk_count,
        issues=issues,
    )
