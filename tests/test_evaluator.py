from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from crag_ingestion.config import IngestionConfig
from crag_ingestion.embeddings.base import Embedder
from crag_ingestion.models import Chunk, SearchResult
from crag_ingestion.pipeline import IngestionPipeline
from crag_ingestion.retrieval import (
    BGEReranker, CorrectiveAction, GeminiRetrievalEvaluator, RerankedHit,
    Relevance,
)
from crag_ingestion.retrieval.evaluator import GEMINI_API_BASE


def _hit(chunk_id: str, text: str) -> RerankedHit:
    chunk = Chunk(chunk_id, "document", 0, text, len(text))
    return RerankedHit(SearchResult(0.8, chunk, "/source.txt", "/raw.txt"), 2.0)


def _response(rows: list[dict[str, str]], *, finish_reason: str = "STOP") -> dict[str, object]:
    return {
        "candidates": [{
            "finishReason": finish_reason,
            "content": {"parts": [{"text": json.dumps({"judgments": rows})}]},
        }]
    }


def test_gemini_evaluator_batches_candidates_and_routes_correct() -> None:
    calls: list[tuple[dict[str, object], str]] = []

    def transport(payload: dict[str, object], key: str) -> dict[str, object]:
        calls.append((payload, key))
        return _response([
            {"chunk_id": "a", "label": "irrelevant", "reason": "No answer."},
            {"chunk_id": "b", "label": "relevant", "reason": "Contains the answer."},
        ])

    evaluator = GeminiRetrievalEvaluator(api_key="dummy-test-key", transport=transport)
    decision = evaluator.evaluate("Who wrote it?", [_hit("a", "Unrelated"), _hit("b", "It was Jane")])
    assert decision.action == CorrectiveAction.CORRECT
    assert [item.relevance for item in decision.hits] == [
        Relevance.IRRELEVANT, Relevance.RELEVANT
    ]
    assert decision.to_dict()["hits"][1]["rerank_score"] == 2.0
    assert len(calls) == 1
    assert calls[0][1] == "dummy-test-key"
    assert decision.model == "gemini-3.5-flash-lite"
    assert calls[0][0]["generationConfig"]["responseMimeType"] == "application/json"
    assert calls[0][0]["generationConfig"]["responseSchema"]["required"] == ["judgments"]
    assert len(json.loads(calls[0][0]["contents"][0]["parts"][0]["text"])["passages"]) == 2


@pytest.mark.parametrize(
    ("labels", "action"),
    [
        (["irrelevant", "irrelevant"], CorrectiveAction.INCORRECT),
        (["irrelevant", "uncertain"], CorrectiveAction.AMBIGUOUS),
        (["relevant", "uncertain"], CorrectiveAction.CORRECT),
    ],
)
def test_gemini_evaluator_routes_from_per_chunk_labels(
    labels: list[str], action: CorrectiveAction,
) -> None:
    rows = [
        {"chunk_id": chunk_id, "label": label, "reason": "Checked passage."}
        for chunk_id, label in zip(("a", "b"), labels, strict=True)
    ]
    evaluator = GeminiRetrievalEvaluator(
        api_key="dummy-test-key", transport=lambda _payload, _key: _response(rows)
    )
    assert evaluator.evaluate("question", [_hit("a", "one"), _hit("b", "two")]).action == action


def test_empty_retrieval_routes_incorrect_without_api_key_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    evaluator = GeminiRetrievalEvaluator()
    assert evaluator.evaluate("question", []).action == CorrectiveAction.INCORRECT


def test_missing_key_is_reported_without_using_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("crag_ingestion.retrieval.evaluator.load_dotenv", lambda: None)
    evaluator = GeminiRetrievalEvaluator(
        transport=lambda _payload, _key: pytest.fail("Network transport must not run")
    )
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        evaluator.evaluate("question", [_hit("a", "passage")])


def test_gemini_request_uses_correct_endpoint_and_api_key_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, *, timeout: int) -> BytesIO:
        assert request.full_url == f"{GEMINI_API_BASE}/gemini-3.5-flash-lite:generateContent"
        assert request.get_header("X-goog-api-key") == "dummy-test-key"
        assert request.get_header("Authorization") is None
        assert request.get_header("Content-type") == "application/json"
        assert timeout == 45
        payload = json.loads(request.data)
        assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"
        assert "dummy-test-key" not in json.dumps(payload)
        return BytesIO(json.dumps(_response([
            {"chunk_id": "a", "label": "relevant", "reason": "Evidence found."}
        ])).encode("utf-8"))

    monkeypatch.setattr("crag_ingestion.retrieval.evaluator.urlopen", fake_urlopen)
    decision = GeminiRetrievalEvaluator(api_key="dummy-test-key").evaluate(
        "question", [_hit("a", "passage")]
    )
    assert decision.action == CorrectiveAction.CORRECT


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (403, "PERMISSION_DENIED", "denied this request"),
        (401, None, "rejected the API key"),
        (429, None, "quota or rate limit"),
        (404, None, "model was not found"),
        (400, None, "rejected the request"),
    ],
)
def test_gemini_http_errors_are_actionable_without_leaking_response_text(
    monkeypatch: pytest.MonkeyPatch, status: int, code: str | None, expected: str,
) -> None:
    body = json.dumps({"error": {"code": code, "message": "private passage and secret"}})

    def fake_urlopen(_request: object, *, timeout: int) -> None:
        raise HTTPError(GEMINI_API_BASE, status, "denied", {}, BytesIO(body.encode("utf-8")))

    monkeypatch.setattr("crag_ingestion.retrieval.evaluator.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match=expected) as captured:
        GeminiRetrievalEvaluator(api_key="dummy-test-key").evaluate(
            "question", [_hit("a", "passage")]
        )
    assert "private passage" not in str(captured.value)
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    "response",
    [
        _response([{"chunk_id": "a", "label": "relevant", "reason": "ok"}], finish_reason="MAX_TOKENS"),
        _response([{"chunk_id": "wrong", "label": "relevant", "reason": "ok"}]),
        _response([{"chunk_id": "a", "label": "maybe", "reason": "ok"}]),
        _response([{"chunk_id": "a", "label": "relevant", "reason": ""}]),
        {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not json"}]}}]},
        {"candidates": []},
    ],
)
def test_invalid_api_response_fails_closed(response: dict[str, object]) -> None:
    evaluator = GeminiRetrievalEvaluator(
        api_key="dummy-test-key", transport=lambda _payload, _key: response
    )
    with pytest.raises(ValueError):
        evaluator.evaluate("question", [_hit("a", "passage")])


def test_pipeline_connects_reranked_hits_to_gemini_evaluator(
    tmp_path: Path, fake_embedder: Embedder,
) -> None:
    source = tmp_path / "facts.txt"
    source.write_text("Chính sách hoàn tiền trong 30 ngày.", encoding="utf-8")

    class FakeCrossEncoder:
        def compute_score(self, pairs: list[list[str]], *, normalize: bool) -> list[float]:
            assert not normalize
            return [3.0 for _ in pairs]

    evaluator = GeminiRetrievalEvaluator(
        api_key="dummy-test-key",
        transport=lambda _payload, _key: _response([
            {"chunk_id": passage["chunk_id"], "label": "relevant", "reason": "Policy found."}
            for passage in json.loads(_payload["contents"][0]["parts"][0]["text"])["passages"]
        ]),
    )
    with IngestionPipeline(
        IngestionConfig(data_dir=tmp_path / "data"),
        embedder=fake_embedder,
        reranker=BGEReranker(model=FakeCrossEncoder()),
        evaluator=evaluator,
    ) as pipeline:
        pipeline.ingest_file(source)
        decision = pipeline.evaluate_retrieval(
            "hoàn tiền", candidate_limit=5, evaluation_limit=1
        )
        assert decision.action == CorrectiveAction.CORRECT
        assert decision.hits[0].hit.rerank_score == 3.0
