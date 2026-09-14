from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .config import IngestionConfig
from .documents import DocumentManager
from .exceptions import IngestionError
from .pipeline import IngestionPipeline
from .utils import sha256_file
from .workflow import CragRunResult, CragWorkflow


class LocalWebService:
    """Synchronous application service; lock the shared local Qdrant client."""

    def __init__(self, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline
        self.documents = DocumentManager(pipeline)
        self.lock = threading.RLock()

    def ask(self, data: dict[str, object]) -> dict[str, object]:
        question = data.get("question")
        candidate_limit = data.get("candidate_limit", 10)
        limit = data.get("limit", 3)
        document_id = data.get("document_id")
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 5000:
            raise ValueError("Question must contain 1–5000 characters")
        if (
            type(candidate_limit) is not int or type(limit) is not int
            or not 1 <= limit <= candidate_limit <= 100
        ):
            raise ValueError("Retrieval limits must satisfy 1 <= limit <= candidate_limit <= 100")
        if document_id is not None:
            if not isinstance(document_id, str):
                raise ValueError("Invalid document ID")
            self.documents.get_document(document_id)
        with self.lock:
            result = self.pipeline.run_crag(
                question.strip(), candidate_limit=candidate_limit,
                evaluation_limit=limit, document_id=document_id,
            )
        return self.summarize(result)

    @staticmethod
    def summarize(result: CragRunResult) -> dict[str, object]:
        strips = {strip.strip_id: strip for strip in result.context.strips}
        used = result.answer.citations if result.answer else result.context.citations
        citations = []
        for citation in used:
            metadata = citation.metadata
            viewer_url: str | None = None
            if citation.source_type == "internal":
                stored_path = metadata.get("stored_path")
                if isinstance(stored_path, str) and stored_path:
                    number = citation.marker.strip("[]")
                    viewer_url = f"/api/runs/{result.run_id}/citations/{number}/source"
                    pages = metadata.get("pages")
                    if isinstance(pages, list) and pages and type(pages[0]) is int:
                        viewer_url += f"#page={pages[0]}"
            elif citation.source_type == "web":
                url = metadata.get("url", citation.source_ref)
                if isinstance(url, str) and urlsplit(url).scheme in {"http", "https"}:
                    viewer_url = url
            strip = strips.get(citation.strip_id)
            citations.append({
                **citation.to_dict(),
                "text": strip.text if strip else "",
                "viewer_url": viewer_url,
            })
        return {
            "run_id": result.run_id,
            "branch": result.decision.action.value,
            "status": result.context.status,
            "answer": result.answer.to_dict() if result.answer else None,
            "citations": citations,
            "warnings": list(result.warnings),
        }

    def get_run(self, run_id: str) -> dict[str, object]:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Invalid run ID")
        return self.summarize(CragWorkflow.load_run(self.pipeline.config.checkpoint_path, run_id))

    def citation_source_file(self, run_id: str, number: str) -> tuple[Path, str]:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id) or not re.fullmatch(r"[1-9][0-9]*", number):
            raise ValueError("Invalid citation reference")
        result = CragWorkflow.load_run(self.pipeline.config.checkpoint_path, run_id)
        marker = f"[{number}]"
        citations = result.answer.citations if result.answer else result.context.citations
        citation = next((item for item in citations if item.marker == marker), None)
        if citation is None or citation.source_type != "internal":
            raise KeyError("Internal citation was not found")
        document_id = citation.metadata.get("document_id")
        stored_path = citation.metadata.get("stored_path")
        if not isinstance(document_id, str) or not re.fullmatch(r"[0-9a-f]{24}", document_id):
            raise ValueError("Citation has no valid document provenance")
        if not isinstance(stored_path, str):
            raise ValueError("Citation has no stored source path")
        path = Path(stored_path)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Citation source copy is missing")
        resolved = path.resolve()
        raw_root = self.pipeline.config.raw_dir.resolve()
        if (
            not resolved.is_relative_to(raw_root)
            or resolved.parent != raw_root / document_id
            or not re.fullmatch(r"[0-9a-f]{64}", resolved.stem)
            or sha256_file(resolved) != resolved.stem
        ):
            raise ValueError("Citation source copy is outside the verified raw store")
        return resolved, mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"

    def upload(self, data: dict[str, object], document_id: str | None = None) -> dict[str, object]:
        filename = data.get("filename")
        encoded = data.get("content_base64")
        if not isinstance(filename, str) or not isinstance(encoded, str):
            raise ValueError("Upload requires filename and content_base64")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Upload content is not valid base64") from None
        with self.lock:
            if document_id is None:
                return self.documents.upload(filename, content)
            return self.documents.update_upload(document_id, filename, content)

    def refresh(self, document_id: str) -> dict[str, object]:
        with self.lock:
            return self.documents.refresh(document_id)

    def delete(self, document_id: str) -> dict[str, object]:
        with self.lock:
            return self.documents.delete(document_id)


class CRAGHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: LocalWebService) -> None:
        if address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("The unauthenticated web app may bind only to localhost")
        self.service = service
        super().__init__(address, CRAGRequestHandler)


class CRAGRequestHandler(BaseHTTPRequestHandler):
    server: CRAGHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        try:
            if not self._allowed_host():
                self._json(403, {"error": "Request host is not local"})
                return
            path = unquote(urlsplit(self.path).path)
            service = self.server.service
            if method == "GET" and path in {"/", "/app.js", "/style.css"}:
                name = "index.html" if path == "/" else path.lstrip("/")
                self._static(name)
                return
            if method == "GET" and path == "/api/documents":
                with service.lock:
                    self._json(200, {"documents": service.documents.list_documents()})
                return
            doc_match = re.fullmatch(r"/api/documents/([0-9a-f]{24})(?:/(source|refresh))?", path)
            if doc_match:
                document_id, suffix = doc_match.groups()
                if method == "GET" and suffix == "source":
                    with service.lock:
                        source, media_type = service.documents.source_file(document_id)
                        self._source(source, media_type)
                    return
                if method == "GET" and suffix is None:
                    with service.lock:
                        self._json(200, service.documents.get_document(document_id))
                    return
                if method == "POST" and suffix == "refresh":
                    self._require_mutation()
                    self._json(200, service.refresh(document_id))
                    return
                if method == "PUT" and suffix is None:
                    self._require_mutation()
                    self._json(200, service.upload(self._read_json(), document_id))
                    return
                if method == "DELETE" and suffix is None:
                    self._require_mutation()
                    self._json(200, service.delete(document_id))
                    return
            if method == "POST" and path == "/api/documents":
                self._require_mutation()
                self._json(201, service.upload(self._read_json()))
                return
            if method == "POST" and path == "/api/ask":
                self._require_mutation()
                self._json(200, service.ask(self._read_json()))
                return
            run_match = re.fullmatch(r"/api/runs/([0-9a-f]{32})", path)
            if method == "GET" and run_match:
                self._json(200, service.get_run(run_match.group(1)))
                return
            cited_source = re.fullmatch(
                r"/api/runs/([0-9a-f]{32})/citations/([1-9][0-9]*)/source", path
            )
            if method == "GET" and cited_source:
                source, media_type = service.citation_source_file(*cited_source.groups())
                self._source(source, media_type)
                return
            self._json(404, {"error": "Route was not found"})
        except KeyError as exc:
            self._json(404, {"error": str(exc).strip("'")})
        except FileNotFoundError as exc:
            self._json(404, {"error": str(exc)})
        except (ValueError, IngestionError) as exc:
            self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(503, {"error": str(exc)})
        except Exception:
            self._json(500, {"error": "Internal server error"})

    def _allowed_host(self) -> bool:
        host = self.headers.get("Host", "")
        port = self.server.server_port
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _require_mutation(self) -> None:
        port = self.server.server_port
        allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        origin = self.headers.get("Origin")
        if origin is not None and origin not in allowed:
            raise ValueError("Cross-origin modification is not allowed")
        if self.headers.get("X-CRAG-Local") != "1":
            raise ValueError("Missing local request header")

    def _read_json(self) -> dict[str, object]:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type must be application/json")
        length = self.headers.get("Content-Length")
        maximum = (self.server.service.pipeline.config.max_file_bytes * 4 // 3) + 100_000
        if not length or not length.isdigit() or not 0 < int(length) <= maximum:
            raise ValueError("Request body is empty or too large")
        try:
            value = json.loads(self.rfile.read(int(length)))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Request body must be valid JSON") from None
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name: str) -> None:
        path = Path(__file__).parent / "static" / name
        if not path.is_file():
            self._json(404, {"error": "Asset was not found"})
            return
        body = path.read_bytes()
        media = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send_response(200)
        self._headers(media + ("; charset=utf-8" if media.startswith("text/") or name.endswith(".js") else ""), len(body))
        self.end_headers()
        self.wfile.write(body)

    def _source(self, path: Path, media_type: str) -> None:
        self.send_response(200)
        self._headers(media_type or "application/octet-stream", path.stat().st_size)
        self.send_header(
            "Content-Disposition",
            "inline" if media_type in {"application/pdf", "text/plain", "text/markdown"} else "attachment",
        )
        self.end_headers()
        with path.open("rb") as stream:
            while block := stream.read(64 * 1024):
                self.wfile.write(block)

    def _headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-src 'self'",
        )


def serve(config: IngestionConfig, *, port: int = 8000) -> None:
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    with IngestionPipeline(config) as pipeline:
        with CRAGHTTPServer(("127.0.0.1", port), LocalWebService(pipeline)) as server:
            print(f"CRAG web app: http://127.0.0.1:{server.server_port}/", flush=True)
            server.serve_forever()
