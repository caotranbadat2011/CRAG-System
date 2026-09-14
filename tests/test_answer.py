from __future__ import annotations

import json

import pytest

from crag_ingestion.config import AnswerConfig
from crag_ingestion.retrieval.answer import (
    GeminiAnswerGenerator, NO_ANSWER, PARTIAL_NOTICE,
)
from crag_ingestion.retrieval.context import AssembledContext, Citation
from crag_ingestion.retrieval.diversity import KnowledgeStrip


def _context(status: str = "ready") -> AssembledContext:
    if status == "no_evidence":
        return AssembledContext(status, "", (), (), ())
    strip = KnowledgeStrip("s1", "Tokyo là thủ đô Nhật Bản.", "web", "https://example.org", {})
    citation = Citation("[1]", "s1", "web", "https://example.org", {})
    line = json.dumps({
        "citation": "[1]", "source_type": "web",
        "source_ref": "https://example.org", "text": strip.text,
    }, ensure_ascii=False)
    return AssembledContext(status, line, (strip,), (citation,), ())


def _response(answerable: bool, claims: list[dict[str, object]]) -> dict[str, object]:
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{
        "text": json.dumps({"answerable": answerable, "claims": claims}, ensure_ascii=False)
    }]}}]}


def test_answer_uses_existing_citation_and_default_model() -> None:
    def transport(payload: dict[str, object], key: str) -> dict[str, object]:
        assert key == "test-key"
        assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}  # type: ignore[index]
        assert payload["generationConfig"]["maxOutputTokens"] == 4096  # type: ignore[index]
        assert "At most 12 claims" in payload["systemInstruction"]["parts"][0]["text"]  # type: ignore[index]
        request = json.loads(payload["contents"][0]["parts"][0]["text"])  # type: ignore[index]
        assert request["evidence"] == _context().text
        return _response(True, [{"text": "Tokyo là thủ đô Nhật Bản.", "citations": ["[1]"]}])

    result = GeminiAnswerGenerator(api_key="test-key", transport=transport).generate(
        "Thủ đô Nhật Bản là gì?", _context()
    )
    assert result.status == "answered"
    assert result.model == "gemini-3.5-flash-lite"
    assert result.text == "Tokyo là thủ đô Nhật Bản. [1]"
    assert [citation.source_ref for citation in result.citations] == ["https://example.org"]


def test_partial_answer_discloses_missing_evidence() -> None:
    generator = GeminiAnswerGenerator(
        api_key="test-key",
        transport=lambda _payload, _key: _response(
            True, [{"text": "Tokyo là thủ đô Nhật Bản.", "citations": ["[1]"]}]
        ),
    )
    answer = generator.generate("Thủ đô Nhật Bản?", _context("partial"))
    assert answer.status == "partial"
    assert answer.text.startswith(PARTIAL_NOTICE)
    assert answer.text.endswith("Tokyo là thủ đô Nhật Bản. [1]")


def test_no_evidence_skips_api_even_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    generator = GeminiAnswerGenerator(transport=lambda *_args: pytest.fail("must not call API"))
    answer = generator.generate("Thủ đô Nhật Bản?", _context("no_evidence"))
    assert answer.status == "insufficient_evidence"
    assert answer.text == NO_ANSWER
    assert answer.citations == () and answer.model is None


@pytest.mark.parametrize("claims", [
    [{"text": "Tokyo là thủ đô.", "citations": ["[99]"]}],
    [{"text": "Tokyo là thủ đô.", "citations": []}],
    [{"text": "Tokyo là thủ đô. [99]", "citations": ["[1]"]}],
    [{"text": "Tokyo là thủ đô.", "citations": ["[1]", "[1]"]}],
])
def test_answer_rejects_invalid_citation(claims: list[dict[str, object]]) -> None:
    generator = GeminiAnswerGenerator(
        api_key="test-key", transport=lambda _payload, _key: _response(True, claims)
    )
    with pytest.raises(ValueError, match="citation|claim text"):
        generator.generate("Thủ đô Nhật Bản?", _context())


def test_answer_can_abstain_with_available_but_irrelevant_evidence() -> None:
    generator = GeminiAnswerGenerator(
        api_key="test-key", transport=lambda _payload, _key: _response(False, [])
    )
    answer = generator.generate("Thủ đô Nhật Bản?", _context())
    assert answer.status == "insufficient_evidence" and answer.text == NO_ANSWER


def test_answer_rejects_incomplete_and_contradictory_responses() -> None:
    generator = GeminiAnswerGenerator(
        api_key="test-key", transport=lambda _payload, _key: {
            "candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]
        },
    )
    with pytest.raises(ValueError, match="incomplete"):
        generator.generate("Thủ đô Nhật Bản?", _context())
    generator = GeminiAnswerGenerator(
        api_key="test-key", transport=lambda _payload, _key: _response(True, [])
    )
    with pytest.raises(ValueError, match="conflicts"):
        generator.generate("Thủ đô Nhật Bản?", _context())


def test_answer_length_settings_are_enforced() -> None:
    with pytest.raises(ValueError, match="max_claims"):
        AnswerConfig(max_claims=0)
    with pytest.raises(ValueError, match="max_output_tokens"):
        AnswerConfig(max_output_tokens=200)

    claims = [{"text": f"Chi tiết {index}.", "citations": ["[1]"]} for index in range(3)]
    generator = GeminiAnswerGenerator(
        api_key="test-key", config=AnswerConfig(max_claims=2, max_output_tokens=1024),
        transport=lambda payload, _key: (
            _response(True, claims)
            if payload["generationConfig"]["maxOutputTokens"] == 1024  # type: ignore[index]
            else pytest.fail("Wrong token budget")
        ),
    )
    with pytest.raises(ValueError, match="invalid claims"):
        generator.generate("Thủ đô Nhật Bản?", _context())


def test_answer_retries_corrupted_diacritics_once() -> None:
    calls: list[dict[str, object]] = []

    def transport(payload: dict[str, object], _key: str) -> dict[str, object]:
        calls.append(payload)
        text = "Tokyo m´ tả thủ đô." if len(calls) == 1 else "Tokyo mô tả thủ đô."
        return _response(True, [{"text": text, "citations": ["[1]"]}])

    answer = GeminiAnswerGenerator(api_key="test-key", transport=transport).generate(
        "Thủ đô Nhật Bản?", _context()
    )
    assert len(calls) == 2
    assert answer.text == "Tokyo mô tả thủ đô. [1]"


def test_answer_rejects_persistently_corrupted_unicode() -> None:
    generator = GeminiAnswerGenerator(
        api_key="test-key",
        transport=lambda *_: _response(True, [{"text": "m´ tả", "citations": ["[1]"]}]),
    )
    with pytest.raises(ValueError, match="corrupted Unicode"):
        generator.generate("Thủ đô Nhật Bản?", _context())
