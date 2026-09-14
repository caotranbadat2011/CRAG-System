from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from crag_ingestion.config import IngestionConfig
from crag_ingestion.pipeline import IngestionPipeline
from crag_ingestion.retrieval import CorrectiveAction, EvaluationDecision, GeneratedAnswer, KnowledgeStrip
from crag_ingestion.retrieval.context import AssembledContext, Citation
from crag_ingestion.web import CRAGHTTPServer, LocalWebService
from crag_ingestion.workflow import CragRunResult


def _request(base: str, path: str, method: str = "GET", payload: dict[str, object] | None = None,
             *, local: bool = True, origin: str | None = None) -> tuple[int, dict[str, object]]:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if method in {"POST", "PUT"} else None
    headers = {"Content-Type": "application/json"}
    if local:
        headers["X-CRAG-Local"] = "1"
    if origin:
        headers["Origin"] = origin
    request = Request(base + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def test_local_api_and_document_lifecycle(
    tmp_path: Path, fake_embedder: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = IngestionConfig(data_dir=tmp_path / "data")
    with IngestionPipeline(config, embedder=fake_embedder) as pipeline:  # type: ignore[arg-type]
        service = LocalWebService(pipeline)
        with CRAGHTTPServer(("127.0.0.1", 0), service) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/", timeout=10) as response:
                    assert b"CRAG System" in response.read()
                with urlopen(base + "/app.js", timeout=10) as response:
                    assert b"renderRun" in response.read()
                status, body = _request(base, "/api/documents")
                assert status == 200 and body["documents"] == []
                status, _ = _request(base, "/api/documents", "POST", {
                    "filename": "file.md", "content_base64": base64.b64encode(b"# Heading\nContent here.").decode(),
                }, local=False)
                assert status == 400
                status, _ = _request(base, "/api/documents", "POST", {
                    "filename": "file.md", "content_base64": base64.b64encode(b"# Heading\nContent here.").decode(),
                }, origin="https://evil.example")
                assert status == 400
                status, added = _request(base, "/api/documents", "POST", {
                    "filename": "file.md", "content_base64": base64.b64encode(b"# Heading\nContent here.").decode(),
                })
                assert status == 201 and added["status"] == "indexed"
                doc_id = added["document_id"]
                status, listed = _request(base, "/api/documents")
                assert status == 200 and listed["documents"][0]["managed_upload"] is True
                with urlopen(base + f"/api/documents/{doc_id}/source", timeout=10) as response:
                    assert response.read() == b"# Heading\nContent here."

                original_raw = pipeline.index.get_document(doc_id)["stored_path"]
                strip = KnowledgeStrip(
                    "strip-1", "Content here.", "internal", "chunk-1",
                    {"document_id": doc_id, "pages": [1], "stored_path": original_raw},
                )
                citation = Citation("[1]", "strip-1", "internal", "chunk-1", strip.metadata)
                run = CragRunResult(
                    "a" * 32, str(config.checkpoint_path),
                    EvaluationDecision(CorrectiveAction.CORRECT, "test", ()),
                    (strip,), (), (), (),
                    AssembledContext("ready", "", (strip,), (citation,), ()),
                    GeneratedAnswer("answered", "Content here. [1]", "test", (citation,)), (),
                )
                monkeypatch.setattr(pipeline, "run_crag", lambda *_args, **_kwargs: run)
                monkeypatch.setattr("crag_ingestion.web.CragWorkflow.load_run", lambda *_args: run)
                status, answer = _request(base, "/api/ask", "POST", {"question": "What content?"})
                assert status == 200 and answer["answer"]["text"] == "Content here. [1]"
                assert answer["citations"][0]["text"] == "Content here."
                assert answer["citations"][0]["viewer_url"].endswith(
                    f"/api/runs/{'a' * 32}/citations/1/source#page=1"
                )
                status, old = _request(base, "/api/runs/" + "a" * 32)
                assert status == 200 and old["run_id"] == "a" * 32

                status, updated = _request(base, f"/api/documents/{doc_id}", "PUT", {
                    "filename": "new.md", "content_base64": base64.b64encode(b"# Changed\nNew evidence.").decode(),
                })
                assert status == 200 and updated["document_id"] == doc_id
                with urlopen(base + f"/api/runs/{'a' * 32}/citations/1/source", timeout=10) as response:
                    assert response.read() == b"# Heading\nContent here."
                status, deleted = _request(base, f"/api/documents/{doc_id}", "DELETE")
                assert status == 200 and deleted["removed_chunks"] >= 1
                status, _ = _request(base, f"/api/documents/{doc_id}")
                assert status == 404
            finally:
                server.shutdown()
                thread.join(timeout=10)


def test_server_rejects_nonlocal_bind() -> None:
    with pytest.raises(ValueError, match="localhost"):
        CRAGHTTPServer(("0.0.0.0", 0), None)  # type: ignore[arg-type]


def test_web_lists_stale_documents_and_blocks_ask_before_model_calls(
    tmp_path: Path, fake_embedder: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with IngestionPipeline(IngestionConfig(data_dir=tmp_path / "data"), embedder=fake_embedder) as pipeline:  # type: ignore[arg-type]
        source = tmp_path / "source.txt"
        source.write_text("Source about arbitration.", encoding="utf-8")
        document_id = pipeline.ingest_file(source).document_id
        monkeypatch.setattr(pipeline, "PIPELINE_VERSION", "new-parser-version")
        service = LocalWebService(pipeline)
        assert service.documents.get_document(document_id)["index_status"] == "outdated_pipeline"
        with CRAGHTTPServer(("127.0.0.1", 0), service) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                status, result = _request(base, "/api/documents")
                assert status == 200
                assert result["documents"][0]["index_status"] == "outdated_pipeline"
                status, result = _request(base, "/api/ask", "POST", {
                    "question": "What?", "document_id": document_id,
                })
                assert status == 400 and "Làm mới" in result["error"]
                status, refreshed = _request(base, f"/api/documents/{document_id}/refresh", "POST")
                assert status == 200 and refreshed["status"] == "indexed"
                status, result = _request(base, "/api/documents")
                assert status == 200 and result["documents"][0]["index_status"] == "current"
            finally:
                server.shutdown()
                thread.join(timeout=10)
