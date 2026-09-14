from __future__ import annotations

import hashlib
import json
import socket
import sys
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from crag_ingestion.config import RefinementConfig, WebSearchConfig
from crag_ingestion.cli import _config, _parser, main
from crag_ingestion.config import IngestionConfig
from crag_ingestion.pipeline import IngestionPipeline
from crag_ingestion.retrieval import (
    CorrectiveAction, EvaluatedHit, EvaluationDecision, GeminiQueryRewriter,
    Relevance, SafePageFetcher, WebKnowledgeResult, WebKnowledgeSearcher, WebSearchHit,
)
from crag_ingestion.retrieval.web_fetch import FetchedPage, WebPageError, normalize_public_url
from crag_ingestion.retrieval.web_search import DDGSDependencyError, DDGSWebSearchProvider


def _decision(action: CorrectiveAction) -> EvaluationDecision:
    return EvaluationDecision(action, "test/evaluator", ())


class FakeRewriter:
    def __init__(self) -> None:
        self.calls = 0

    def rewrite(self, question: str, *, limit: int) -> list[str]:
        self.calls += 1
        assert question == "hoàn tiền" and limit == 2
        return ["chính sách hoàn tiền"]


class FakeProvider:
    def __init__(self, hits: list[WebSearchHit]) -> None:
        self.hits = hits
        self.calls = 0

    def search(self, query: str, *, limit: int) -> list[WebSearchHit]:
        self.calls += 1
        assert query == "chính sách hoàn tiền"
        return self.hits[:limit]


class FakeFetcher:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        text = self.pages[url]
        return FetchedPage(url, text, hashlib.sha256(text.encode()).hexdigest(), "2026-01-01T00:00:00Z", False)


class FakeReranker:
    name = "test/reranker"

    def score(self, query: str, passages: list[str]) -> list[float]:
        assert query == "hoàn tiền"
        return [2.0 if "hoàn tiền" in passage.casefold() else -2.0 for passage in passages]


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, query: str, hits: list[object]) -> EvaluationDecision:
        assert query == "hoàn tiền"
        self.calls += 1
        judged = tuple(EvaluatedHit(
            hit,
            Relevance.RELEVANT if "hoàn tiền" in hit.result.chunk.text.casefold() else Relevance.IRRELEVANT,
            "evidence checked",
        ) for hit in hits)
        return EvaluationDecision(CorrectiveAction.CORRECT, "test/evaluator", judged)


@pytest.mark.parametrize("action", [CorrectiveAction.INCORRECT, CorrectiveAction.AMBIGUOUS])
def test_web_branch_rewrites_fetches_filters_and_keeps_citations(action: CorrectiveAction) -> None:
    url = "https://example.org/policy"
    text = "Hoàn tiền trong 30 ngày. Cửa hàng mở lúc 9 giờ."
    hit = WebSearchHit("Chính sách", url, "irrelevant snippet", "chính sách hoàn tiền", 0)
    rewriter = FakeRewriter()
    provider = FakeProvider([hit, hit])
    fetcher = FakeFetcher({url: text})
    judge = FakeEvaluator()
    result = WebKnowledgeSearcher(
        rewriter, provider, fetcher, FakeReranker(), judge,
        config=WebSearchConfig(), refinement=RefinementConfig(80, 1, 2),
    ).search("hoàn tiền", _decision(action))

    assert result.action == action
    assert result.queries == ("chính sách hoàn tiền",)
    assert result.source_urls == (url,)
    assert fetcher.calls == [url]
    assert judge.calls == 1
    assert len(result.strips) == 1
    strip = result.strips[0]
    assert strip.text == "Hoàn tiền trong 30 ngày."
    assert strip.source_type == "web" and strip.source_ref == url
    assert strip.metadata["url"] == url
    assert strip.metadata["title"] == "Chính sách"
    assert strip.metadata["strip_relevance"] == "relevant"
    assert text[strip.metadata["page_char_start"]:strip.metadata["page_char_end"]] == strip.text


def test_correct_branch_does_not_search_web() -> None:
    rewriter = FakeRewriter()
    provider = FakeProvider([])
    result = WebKnowledgeSearcher(
        rewriter, provider, FakeFetcher({}), FakeReranker(), FakeEvaluator()
    ).search("hoàn tiền", _decision(CorrectiveAction.CORRECT))
    assert not result.strips and rewriter.calls == 0 and provider.calls == 0


def test_search_skips_disallowed_sources_and_does_not_use_snippets_as_evidence() -> None:
    query = "chính sách hoàn tiền"
    hits = [
        WebSearchHit("Private", "http://127.0.0.1/secret", "hoàn tiền", query, 0),
        WebSearchHit("Other", "https://other.org/", "hoàn tiền", query, 1),
        WebSearchHit("Allowed", "https://example.org/unrelated", "hoàn tiền", query, 2),
    ]
    url = "https://example.org/unrelated"
    fetcher = FakeFetcher({url: "Thông tin về thời tiết hôm nay."})
    result = WebKnowledgeSearcher(
        FakeRewriter(), FakeProvider(hits), fetcher, FakeReranker(), FakeEvaluator(),
        config=WebSearchConfig(allowed_domains=("example.org",)),
    ).search("hoàn tiền", _decision(CorrectiveAction.INCORRECT))
    assert result.strips == ()
    assert result.source_urls == (url,)
    assert len(result.warnings) == 3
    assert "No web passages passed relevance filtering" in result.warnings
    assert any("Source domain is not in the allowlist" in item for item in result.warnings)
    assert fetcher.calls == [url]


def test_lexical_preselection_respects_small_candidate_limit() -> None:
    url = "https://example.org/p"
    text = "\n".join(f"Đoạn thứ {number}." for number in range(12))
    result = WebKnowledgeSearcher(
        FakeRewriter(), FakeProvider([WebSearchHit("A", url, "", "chính sách hoàn tiền", 0)]),
        FakeFetcher({url: text}), FakeReranker(), FakeEvaluator(),
        config=WebSearchConfig(max_candidate_passages=2, passages_per_page=2),
        refinement=RefinementConfig(80, 1, 2),
    ).search("hoàn tiền", _decision(CorrectiveAction.INCORRECT))
    assert result.strips == ()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://localhost/a", "http://127.0.0.1/a",
    "http://10.0.0.1/a", "http://[::1]/a", "http://example.org:8080/a",
    "https://user:pass@example.org/a", "https://server.local/a",
    "https://localhost./a", "https://server.local./a",
])
def test_url_rejects_nonpublic_targets(url: str) -> None:
    with pytest.raises(WebPageError):
        normalize_public_url(url)


def test_url_normalization_and_allowlist() -> None:
    assert normalize_public_url("HTTPS://Docs.Example.org/a#part", ("example.org",)) == "https://docs.example.org/a"
    with pytest.raises(WebPageError):
        normalize_public_url("https://example.com/", ("example.org",))


def test_url_normalization_encodes_unicode_without_double_encoding() -> None:
    url = "https://vi.wikipedia.org/wiki/Thủ_đô_của_Nhật_Bản?q=Thủ+đô&next=%2F"
    expected = (
        "https://vi.wikipedia.org/wiki/Th%E1%BB%A7_%C4%91%C3%B4_c%E1%BB%A7a_"
        "Nh%E1%BA%ADt_B%E1%BA%A3n?q=Th%E1%BB%A7+%C4%91%C3%B4&next=%2F"
    )
    assert normalize_public_url(url) == expected
    assert normalize_public_url(expected) == expected
    expected.encode("ascii")


def test_ddgs_adapter_uses_duckduckgo_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            seen["timeout"] = timeout

        def text(self, query: str, **kwargs: object) -> list[dict[str, str]]:
            seen.update(kwargs)
            return [{"title": "Page", "href": "https://example.org/", "body": "Snippet"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    hits = DDGSWebSearchProvider(region="vn-vi", timeout=7).search("hoàn tiền", limit=3)
    assert seen["backend"] == "duckduckgo"
    assert seen["region"] == "vn-vi" and seen["max_results"] == 3
    assert hits[0].url == "https://example.org/" and hits[0].snippet == "Snippet"


def test_ddgs_retries_empty_href_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    delays: list[float] = []

    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 10

        def text(self, _query: str, **kwargs: object) -> list[dict[str, str]]:
            assert kwargs["backend"] == "duckduckgo"
            calls.append(1)
            if len(calls) == 1:
                return [{"title": "Broken", "href": "", "body": "not evidence"}]
            return [{"title": "Valid", "href": "https://example.org/", "body": "snippet"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    provider = DDGSWebSearchProvider(attempts=3, backoff_seconds=0.25, sleep=delays.append)
    hits = provider.search("hoàn tiền", limit=3)
    assert len(calls) == 2 and delays == [0.25]
    assert [hit.url for hit in hits] == ["https://example.org/"]
    assert provider.last_diagnostics.attempts == 2
    assert provider.last_diagnostics.empty_hrefs == 1
    assert provider.last_diagnostics.invalid_urls == 0


def test_ddgs_retries_exceptions_with_backoff_and_sanitized_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    delays: list[float] = []

    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            pass

        def text(self, _query: str, **_kwargs: object) -> list[dict[str, str]]:
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("secret URL and query must not leak")
            return [{"href": "https://example.org/"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    provider = DDGSWebSearchProvider(attempts=3, backoff_seconds=0.2, sleep=delays.append)
    assert len(provider.search("hoàn tiền", limit=1)) == 1
    assert len(calls) == 3 and delays == [0.2, 0.4]
    assert provider.last_diagnostics.error_types == ("TimeoutError", "TimeoutError")
    assert "secret" not in provider.last_diagnostics.summary()


def test_ddgs_all_invalid_urls_are_bounded_and_never_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            pass

        def text(self, _query: str, **_kwargs: object) -> list[dict[str, str]]:
            calls.append(1)
            return [{"href": ""}, {"href": "file:///private"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    provider = DDGSWebSearchProvider(attempts=3, backoff_seconds=0, sleep=lambda _delay: None)
    assert provider.search("hoàn tiền", limit=2) == []
    assert len(calls) == 3
    assert provider.last_diagnostics.empty_hrefs == 3
    assert provider.last_diagnostics.invalid_urls == 3


def test_ddgs_exhausted_errors_preserve_type_not_raw_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            pass

        def text(self, _query: str, **_kwargs: object) -> object:
            raise RuntimeError("secret URL or query")

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    provider = DDGSWebSearchProvider(attempts=2, backoff_seconds=0, sleep=lambda _delay: None)
    with pytest.raises(RuntimeError, match="DuckDuckGo search failed") as exc:
        provider.search("hoàn tiền", limit=2)
    assert "secret" not in str(exc.value)
    assert provider.last_diagnostics.error_types == ("RuntimeError", "RuntimeError")


def test_web_search_falls_back_to_original_question_without_using_snippets() -> None:
    class Rewriter:
        def rewrite(self, _question: str, *, limit: int) -> list[str]:
            assert limit == 2
            return ["rewritten query"]

    class Provider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search(self, query: str, *, limit: int) -> list[WebSearchHit]:
            self.calls.append(query)
            if query == "rewritten query":
                return [WebSearchHit("Bad", "", "hoàn tiền", query, 0)]
            return [WebSearchHit("Source", "https://example.org/refund", "", query, 0)]

    provider = Provider()
    url = "https://example.org/refund"
    result = WebKnowledgeSearcher(
        Rewriter(), provider, FakeFetcher({url: "Hoàn tiền trong 30 ngày."}),
        FakeReranker(), FakeEvaluator(),
    ).search("hoàn tiền", _decision(CorrectiveAction.INCORRECT))
    assert provider.calls == ["rewritten query", "hoàn tiền"]
    assert result.queries == ("rewritten query", "hoàn tiền")
    assert result.source_urls == (url,) and len(result.strips) == 1
    assert any("original question" in warning for warning in result.warnings)
    assert all("https://example.org/refund" not in warning for warning in result.warnings)


def test_web_search_does_not_repeat_identical_original_query() -> None:
    class Rewriter:
        def rewrite(self, question: str, *, limit: int) -> list[str]:
            return [question]

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, _query: str, *, limit: int) -> list[WebSearchHit]:
            self.calls += 1
            return []

    provider = Provider()
    result = WebKnowledgeSearcher(
        Rewriter(), provider, FakeFetcher({}), FakeReranker(), FakeEvaluator(),
    ).search("hoàn tiền", _decision(CorrectiveAction.INCORRECT))
    assert provider.calls == 1
    assert result.queries == ("hoàn tiền",)
    assert "No eligible DuckDuckGo results" in result.warnings


def test_missing_ddgs_dependency_fails_explicitly() -> None:
    class Rewriter:
        def rewrite(self, _question: str, *, limit: int) -> list[str]:
            return ["rewritten query"]

    class Provider:
        def search(self, _query: str, *, limit: int) -> list[WebSearchHit]:
            raise DDGSDependencyError("ddgs is not installed")

    with pytest.raises(DDGSDependencyError, match="not installed"):
        WebKnowledgeSearcher(
            Rewriter(), Provider(), FakeFetcher({}), FakeReranker(), FakeEvaluator(),
        ).search("hoàn tiền", _decision(CorrectiveAction.INCORRECT))


def test_web_search_warnings_report_error_type_without_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            pass

        def text(self, _query: str, **_kwargs: object) -> object:
            raise TimeoutError("secret request and URL")

    class Rewriter:
        def rewrite(self, _question: str, *, limit: int) -> list[str]:
            return ["rewritten query"]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    provider = DDGSWebSearchProvider(attempts=1, sleep=lambda _delay: None)
    result = WebKnowledgeSearcher(
        Rewriter(), provider, FakeFetcher({}), FakeReranker(), FakeEvaluator(),
    ).search("hoàn tiền", _decision(CorrectiveAction.INCORRECT))
    assert result.source_urls == ()
    assert result.queries == ("rewritten query", "hoàn tiền")
    assert sum("error_types=TimeoutError" in item for item in result.warnings) == 2
    assert all("secret" not in item for item in result.warnings)


def test_cli_exposes_bounded_ddgs_retry_settings() -> None:
    args = _parser().parse_args([
        "--web-search-attempts", "2", "--web-search-backoff", "0.25",
        "search-web", "hoàn tiền",
    ])
    config = _config(args)
    assert config.web_search.search_attempts == 2
    assert config.web_search.search_backoff_seconds == 0.25
    with pytest.raises(ValueError, match="retry"):
        WebSearchConfig(search_attempts=6)


def test_gemini_rewriter_validates_structured_output() -> None:
    def transport(payload: dict[str, object], key: str) -> dict[str, object]:
        assert key == "test-key"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"  # type: ignore[index]
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({
            "queries": ["  chính sách hoàn tiền  ", "chính sách hoàn tiền"]
        })}]}}]}

    assert GeminiQueryRewriter(api_key="test-key", transport=transport).rewrite("hoàn tiền", limit=2) == [
        "chính sách hoàn tiền"
    ]


def test_gemini_rewriter_rejects_malformed_output() -> None:
    rewrite = GeminiQueryRewriter(
        api_key="test-key", transport=lambda _payload, _key: {"candidates": []}
    )
    with pytest.raises(ValueError, match="malformed"):
        rewrite.rewrite("hoàn tiền", limit=2)


def test_safe_fetcher_extracts_text_and_excludes_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))
    ])

    class Response:
        def __init__(self) -> None:
            self.headers = Message()
            self.headers["Content-Type"] = "text/html; charset=utf-8"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, size: int) -> bytes:
            assert size == 1001
            return b"<html><nav>Menu</nav><main><p>Refund in 30 days.</p><script>bad()</script><p>Keep receipt.</p></main></html>"

    fetcher = SafePageFetcher(WebSearchConfig(max_page_bytes=1000))
    monkeypatch.setattr(fetcher._opener, "open", lambda *_args, **_kwargs: Response())
    page = fetcher.fetch("https://example.org/policy")
    assert "Refund in 30 days." in page.text and "Keep receipt." in page.text
    assert "Menu" not in page.text and "bad()" not in page.text


def test_safe_fetcher_rejects_redirect_to_private_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))
    ])
    fetcher = SafePageFetcher()

    def redirect(*_args: object, **_kwargs: object) -> object:
        headers = Message()
        headers["Location"] = "http://127.0.0.1/private"
        raise HTTPError("https://example.org/", 302, "Found", headers, None)

    monkeypatch.setattr(fetcher._opener, "open", redirect)
    with pytest.raises(WebPageError, match="Nonpublic IP"):
        fetcher.fetch("https://example.org/")


def test_safe_fetcher_encodes_unicode_url_and_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))
    ])
    fetcher = SafePageFetcher()
    requested: list[str] = []

    class Response:
        def __init__(self) -> None:
            self.headers = Message()
            self.headers["Content-Type"] = "text/plain; charset=utf-8"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, _size: int) -> bytes:
            return b"Tokyo is the capital of Japan."

    def open_page(request: object, **_kwargs: object) -> Response:
        requested.append(request.full_url)  # type: ignore[attr-defined]
        request.full_url.encode("ascii")  # type: ignore[attr-defined]
        if len(requested) == 1:
            headers = Message()
            headers["Location"] = "/wiki/Thủ_đô?from=%2F"
            raise HTTPError(request.full_url, 302, "Found", headers, None)  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(fetcher._opener, "open", open_page)
    page = fetcher.fetch("https://example.org/Thủ_đô")
    assert requested == [
        "https://example.org/Th%E1%BB%A7_%C4%91%C3%B4",
        "https://example.org/wiki/Th%E1%BB%A7_%C4%91%C3%B4?from=%2F",
    ]
    assert page.url == requested[-1]


def test_safe_fetcher_rejects_private_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", 0))
    ])
    with pytest.raises(WebPageError, match="nonpublic"):
        SafePageFetcher().fetch("https://example.org/")


@pytest.mark.parametrize("content_type,body,error", [
    ("application/pdf", b"%PDF", "not HTML"),
    ("text/plain", b"x" * 101, "byte limit"),
])
def test_safe_fetcher_rejects_nontext_or_oversized_response(
    monkeypatch: pytest.MonkeyPatch, content_type: str, body: bytes, error: str
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))
    ])

    class Response:
        def __init__(self) -> None:
            self.headers = Message()
            self.headers["Content-Type"] = content_type

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, size: int) -> bytes:
            return body[:size]

    fetcher = SafePageFetcher(WebSearchConfig(max_page_bytes=100))
    monkeypatch.setattr(fetcher._opener, "open", lambda *_args, **_kwargs: Response())
    with pytest.raises(WebPageError, match=error):
        fetcher.fetch("https://example.org/")


def test_pipeline_correct_branch_is_lazy(tmp_path: Path, fake_embedder: object) -> None:
    with IngestionPipeline(
        IngestionConfig(data_dir=tmp_path / "runtime"), embedder=fake_embedder  # type: ignore[arg-type]
    ) as pipeline:
        result = pipeline.search_web_knowledge("hoàn tiền", _decision(CorrectiveAction.CORRECT))
        assert result.strips == () and pipeline._reranker is None and pipeline._evaluator is None


def test_pipeline_incorrect_branch_uses_web_search_components(
    tmp_path: Path, fake_embedder: object
) -> None:
    url = "https://example.org/refunds"
    question = "hoàn tiền"
    text = "Hoàn tiền trong 30 ngày."
    with IngestionPipeline(
        IngestionConfig(data_dir=tmp_path / "runtime", web_search=WebSearchConfig(
            allowed_domains=("example.org",)
        )),
        embedder=fake_embedder,  # type: ignore[arg-type]
        reranker=FakeReranker(),
        evaluator=FakeEvaluator(),
    ) as pipeline:
        result = pipeline.search_web_knowledge(
            question, _decision(CorrectiveAction.INCORRECT),
            rewriter=FakeRewriter(),
            provider=FakeProvider([WebSearchHit("Refunds", url, "", "chính sách hoàn tiền", 0)]),
            fetcher=FakeFetcher({url: text}),
        )
    assert len(result.strips) == 1 and result.strips[0].source_ref == url


def test_cli_search_web_routes_evaluated_decision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakePipeline:
        def __init__(self, config: IngestionConfig) -> None:
            assert config.web_search.allowed_domains == ("example.org",)
            assert config.web_search.max_pages == 2

        def __enter__(self) -> "FakePipeline":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def evaluate_retrieval(self, query: str, **kwargs: object) -> EvaluationDecision:
            assert query == "hoàn tiền" and kwargs["evaluation_limit"] == 3
            return _decision(CorrectiveAction.INCORRECT)

        def search_web_knowledge(
            self, query: str, decision: EvaluationDecision
        ) -> WebKnowledgeResult:
            assert query == "hoàn tiền" and decision.action == CorrectiveAction.INCORRECT
            return WebKnowledgeResult(decision.action, ("query",), (), (), ())

    monkeypatch.setattr("crag_ingestion.cli.IngestionPipeline", FakePipeline)
    status = main([
        "--web-allowed-domain", "example.org", "--web-max-pages", "2",
        "search-web", "hoàn tiền", "--limit", "3",
    ])
    output = json.loads(capsys.readouterr().out)
    assert status == 0 and output["decision"]["action"] == "Incorrect"
    assert output["queries"] == ["query"] and output["web_strips"] == []
