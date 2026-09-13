from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .index import QdrantVectorIndex
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


def validate_index(index: QdrantVectorIndex, check_files: bool = True) -> ValidationReport:
    issues: list[ValidationIssue] = []
    schema = index.collection_schema()
    if not schema["exists"]:
        return ValidationReport(True, 0, 0, [])
    if schema["distance"] != "Cosine":
        issues.append(
            ValidationIssue(
                "error",
                "qdrant_distance",
                f"Expected Cosine distance, got {schema['distance']}",
            )
        )
    if not schema["hybrid_vectors"]:
        issues.append(
            ValidationIssue(
                "error", "qdrant_hybrid_schema",
                "Collection must have named dense and sparse vectors",
            )
        )
    if schema["status"] not in {"green", "yellow"}:
        issues.append(
            ValidationIssue(
                "error",
                "qdrant_status",
                f"Collection status is {schema['status']}",
            )
        )

    documents = index.documents()
    seen_ordinals: dict[str, list[int]] = {}
    rows = list(index.diagnostic_rows())

    for document in documents:
        document_id = str(document["document_id"])
        actual = index.count_document_chunks(document_id)
        if actual != document["chunk_count"]:
            issues.append(
                ValidationIssue(
                    "error",
                    "chunk_count_mismatch",
                    f"metadata={document['chunk_count']}, actual={actual}",
                    document_id,
                )
            )
        if document["embedding_dimensions"] != schema["vector_size"]:
            issues.append(
                ValidationIssue(
                    "error",
                    "collection_dimensions",
                    f"document={document['embedding_dimensions']}, collection={schema['vector_size']}",
                    document_id,
                )
            )
        if check_files:
            _validate_artifacts(document, issues)

    declared_chunks = sum(int(document["chunk_count"]) for document in documents)
    if schema["points_count"] != len(rows) or declared_chunks != len(rows):
        issues.append(
            ValidationIssue(
                "error",
                "qdrant_point_count",
                f"collection={schema['points_count']}, payload={len(rows)}, declared={declared_chunks}",
            )
        )

    for row in rows:
        document_id = str(row.get("document_id", ""))
        chunk_id = str(row.get("chunk_id", ""))
        try:
            ordinal = int(row["ordinal"])
            seen_ordinals.setdefault(document_id, []).append(ordinal)
            vector = row["vector"]
            dimensions = int(row["embedding_dimensions"])
            if not isinstance(vector, list) or len(vector) != dimensions:
                issues.append(
                    ValidationIssue(
                        "error",
                        "vector_dimensions",
                        f"expected {dimensions}, got {len(vector) if isinstance(vector, list) else 'invalid'}",
                        document_id,
                        chunk_id,
                    )
                )
            else:
                norm = math.sqrt(sum(float(value) * float(value) for value in vector))
                if not math.isfinite(norm) or not 0.98 <= norm <= 1.02:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "vector_norm",
                            f"L2 norm is {norm:.6f}",
                            document_id,
                            chunk_id,
                        )
                    )
            sparse = row.get("sparse_vector")
            if not isinstance(sparse, dict):
                issues.append(ValidationIssue(
                    "error", "sparse_vector_missing", "Sparse vector is missing",
                    document_id, chunk_id,
                ))
            else:
                indices = sparse.get("indices")
                values = sparse.get("values")
                if (
                    not isinstance(indices, list) or not isinstance(values, list)
                    or not indices or len(indices) != len(values)
                    or len(indices) != len(set(indices))
                    or any(not isinstance(value, int) or value < 0 for value in indices)
                    or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0
                        for value in values
                    )
                ):
                    issues.append(ValidationIssue(
                        "error", "sparse_vector_invalid",
                        "Sparse token IDs and lexical weights are invalid",
                        document_id, chunk_id,
                    ))
            text = row["text"]
            if not isinstance(text, str) or int(row["char_count"]) != len(text):
                issues.append(
                    ValidationIssue(
                        "error",
                        "char_count",
                        "Stored char count is incorrect",
                        document_id,
                        chunk_id,
                    )
                )
            pages = row.get("pages")
            headings = row.get("heading_path")
            metadata = row.get("metadata")
            if not isinstance(pages, list) or not all(
                isinstance(page, int) and page > 0 for page in pages
            ):
                raise ValueError("pages must be positive integers")
            if not isinstance(headings, list) or not all(
                isinstance(value, str) for value in headings
            ):
                raise ValueError("heading path must contain strings")
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            if row.get("media_type") == "application/pdf" and not pages:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "pdf_page_missing",
                        "PDF chunk has no page reference",
                        document_id,
                        chunk_id,
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    "error", "invalid_payload", str(exc), document_id or None, chunk_id or None
                )
            )

    for document_id, ordinals in seen_ordinals.items():
        ordered = sorted(ordinals)
        if ordered != list(range(len(ordered))):
            issues.append(
                ValidationIssue(
                    "error", "ordinal_gap", f"ordinals={ordered}", document_id
                )
            )

    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        document_count=len(documents),
        chunk_count=len(rows),
        issues=issues,
    )


def _validate_artifacts(
    document: dict[str, object], issues: list[ValidationIssue]
) -> None:
    document_id = str(document["document_id"])
    for field_name in ("stored_path", "processed_path"):
        artifact_path = Path(str(document[field_name]))
        if not artifact_path.is_file():
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_artifact",
                    f"Missing {field_name}: {artifact_path}",
                    document_id,
                )
            )
    stored_path = Path(str(document["stored_path"]))
    if stored_path.is_file() and sha256_file(stored_path) != document["content_hash"]:
        issues.append(
            ValidationIssue(
                "error",
                "raw_checksum",
                "Stored source checksum does not match the Qdrant payload",
                document_id,
            )
        )
    processed_path = Path(str(document["processed_path"]))
    if not processed_path.is_file():
        return
    try:
        processed = json.loads(processed_path.read_text(encoding="utf-8"))
        expected = {
            "document_id": document_id,
            "content_hash": document["content_hash"],
            "pipeline_signature": document["pipeline_signature"],
        }
        for key, expected_value in expected.items():
            if processed.get(key) != expected_value:
                raise ValueError(f"processed {key} does not match the Qdrant payload")
        if len(processed.get("chunks", [])) != document["chunk_count"]:
            raise ValueError("processed chunk count does not match the Qdrant payload")
        if document["media_type"] == "text/markdown":
            _validate_markdown_provenance(processed, str(document["source_path"]))
        if document["media_type"] == "application/pdf":
            _validate_pdf_provenance(processed, str(document["source_path"]))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        issues.append(
            ValidationIssue("error", "processed_artifact", str(exc), document_id)
        )


def _validate_markdown_provenance(processed: dict[str, object], source_path: str) -> None:
    for block_index, block in enumerate(processed.get("blocks", [])):
        provenance = block.get("metadata", {}).get("source")
        if not isinstance(provenance, dict):
            raise ValueError(f"Markdown block {block_index} has no source provenance")
        required = {"path", "start_line", "end_line", "start_char", "end_char"}
        if not required.issubset(provenance):
            raise ValueError(f"Markdown block {block_index} has incomplete source provenance")
        if provenance["path"] != source_path:
            raise ValueError(f"Markdown block {block_index} points to a different source")
        if not (
            1 <= provenance["start_line"] <= provenance["end_line"]
            and 0 <= provenance["start_char"] <= provenance["end_char"]
        ):
            raise ValueError(f"Markdown block {block_index} has an invalid source range")


def _validate_pdf_provenance(processed: dict[str, object], source_path: str) -> None:
    for block_index, block in enumerate(processed.get("blocks", [])):
        provenance = block.get("metadata", {}).get("source")
        if not isinstance(provenance, dict):
            raise ValueError(f"PDF block {block_index} has no source provenance")
        bbox = provenance.get("bbox")
        if (
            provenance.get("path") != source_path
            or not isinstance(provenance.get("page"), int)
            or provenance["page"] < 1
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            raise ValueError(f"PDF block {block_index} has invalid source provenance")
        image = block.get("metadata", {}).get("image")
        if isinstance(image, dict) and image.get("extracted_path"):
            image_path = Path(image["extracted_path"])
            if not image_path.is_file():
                raise ValueError(f"PDF image artifact is missing: {image_path}")
            if sha256_file(image_path) != image.get("sha256"):
                raise ValueError(f"PDF image checksum does not match: {image_path}")
