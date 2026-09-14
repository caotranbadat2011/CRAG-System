from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.parse import urlsplit

from ..config import RefinementConfig, WebSearchConfig
from ..models import Chunk, SearchResult
from .diversity import KnowledgeStrip
from .evaluator import CorrectiveAction, EvaluationDecision, Relevance, RetrievalEvaluator
from .query_rewrite import QueryRewriter
from .refinement import KnowledgeRefiner
from .reranker import Reranker
from .service import RerankedHit
from .web_fetch import FetchedPage, WebPageError, normalize_public_url


@dataclass(frozen=True, slots=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str
    query: str
    rank: int


class WebSearchProvider(Protocol):
    def search(self, query: str, *, limit: int) -> list[WebSearchHit]: ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


@dataclass(frozen=True, slots=True)
class DDGSDiagnostics:
    attempts: int = 0
    empty_hrefs: int = 0
    invalid_urls: int = 0
    error_types: tuple[str, ...] = ()

    def summary(self) -> str:
        details: list[str] = []
        if self.attempts > 1:
            details.append(f"attempts={self.attempts}")
        if self.empty_hrefs:
            details.append(f"empty_hrefs={self.empty_hrefs}")
        if self.invalid_urls:
            details.append(f"invalid_urls={self.invalid_urls}")
        if self.error_types:
            details.append(f"error_types={','.join(dict.fromkeys(self.error_types))}")
        return ", ".join(details)


class DDGSDependencyError(RuntimeError):
    """The optional web-search dependency is unavailable."""


class DDGSWebSearchProvider:
    """DuckDuckGo-only text search through the ddgs package."""

    def __init__(
        self,
        *,
        region: str = "vn-vi",
        timeout: int = 10,
        attempts: int = 3,
        backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not 1 <= attempts <= 5 or not 0 <= backoff_seconds <= 5:
            raise ValueError("DDGS retry settings are out of range")
        self.region = region
        self.timeout = timeout
        self.attempts = attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep or time.sleep
        self.last_diagnostics = DDGSDiagnostics()

    def search(self, query: str, *, limit: int) -> list[WebSearchHit]:
        if not query.strip() or limit <= 0:
            raise ValueError("DDGS search requires a query and positive limit")
        self.last_diagnostics = DDGSDiagnostics()
        try:
            from ddgs import DDGS
        except ImportError:
            raise DDGSDependencyError("ddgs is not installed; run pip install -e .") from None
        empty_hrefs = invalid_urls = 0
        error_types: list[str] = []
        for attempt in range(1, self.attempts + 1):
            hits: list[WebSearchHit] = []
            try:
                rows = DDGS(timeout=self.timeout).text(
                    query, region=self.region, safesearch="moderate",
                    max_results=limit, backend="duckduckgo",
                )
                for rank, row in enumerate(rows or []):
                    if not isinstance(row, dict):
                        invalid_urls += 1
                        continue
                    href = row.get("href")
                    if not isinstance(href, str) or not href.strip():
                        empty_hrefs += 1
                        continue
                    try:
                        normalize_public_url(href)
                    except WebPageError:
                        invalid_urls += 1
                        continue
                    hits.append(WebSearchHit(
                        title=str(row.get("title") or ""),
                        url=href,
                        snippet=str(row.get("body") or ""),
                        query=query,
                        rank=rank,
                    ))
            except Exception as exc:
                # Do not expose DDGS exception messages: they may contain URLs or query text.
                error_types.append(type(exc).__name__)
            self.last_diagnostics = DDGSDiagnostics(
                attempt, empty_hrefs, invalid_urls, tuple(error_types),
            )
            if hits:
                return hits[:limit]
            if attempt < self.attempts:
                self._sleep(self.backoff_seconds * 2 ** (attempt - 1))
        if error_types:
            raise RuntimeError("DuckDuckGo search failed or returned no usable URLs") from None
        return []


@dataclass(frozen=True, slots=True)
class WebKnowledgeResult:
    action: CorrectiveAction
    queries: tuple[str, ...]
    source_urls: tuple[str, ...]
    strips: tuple[KnowledgeStrip, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "queries": list(self.queries),
            "source_urls": list(self.source_urls),
            "web_strips": [strip.to_dict() for strip in self.strips],
            "warnings": list(self.warnings),
        }


class WebKnowledgeSearcher:
    """Rewrite, search, fetch, rank, and verify external evidence for CRAG."""

    def __init__(
        self,
        rewriter: QueryRewriter,
        provider: WebSearchProvider,
        fetcher: PageFetcher,
        reranker: Reranker,
        evaluator: RetrievalEvaluator,
        *,
        config: WebSearchConfig | None = None,
        refinement: RefinementConfig | None = None,
    ) -> None:
        self.rewriter = rewriter
        self.provider = provider
        self.fetcher = fetcher
        self.reranker = reranker
        self.evaluator = evaluator
        self.config = config or WebSearchConfig()
        self.refinement = refinement or RefinementConfig()

    def search(self, question: str, decision: EvaluationDecision) -> WebKnowledgeResult:
        if not question.strip():
            raise ValueError("A nonempty question is required for web knowledge search")
        if decision.action == CorrectiveAction.CORRECT:
            return WebKnowledgeResult(decision.action, (), (), (), ())
        rewritten = self.rewriter.rewrite(question, limit=self.config.max_queries)
        if not rewritten or len(rewritten) > self.config.max_queries or any(not q.strip() for q in rewritten):
            raise ValueError("Query rewriter returned an invalid search query set")

        warnings: list[str] = []
        attempted_queries: list[str] = []
        found: list[tuple[str, WebSearchHit]] = []
        seen: set[str] = set()

        def collect(query: str, label: str) -> None:
            attempted_queries.append(query)
            try:
                hits = self.provider.search(query, limit=self.config.results_per_query)
            except DDGSDependencyError:
                raise
            except RuntimeError:
                diagnostics = getattr(self.provider, "last_diagnostics", None)
                detail = diagnostics.summary() if isinstance(diagnostics, DDGSDiagnostics) else ""
                warnings.append(f"DDGS query {label} failed" + (f": {detail}" if detail else ""))
                return
            diagnostics = getattr(self.provider, "last_diagnostics", None)
            if isinstance(diagnostics, DDGSDiagnostics) and diagnostics.summary():
                warnings.append(f"DDGS query {label}: {diagnostics.summary()}")
            for hit in hits:
                try:
                    url = normalize_public_url(hit.url, self.config.allowed_domains)
                except WebPageError as exc:
                    warnings.append(f"DDGS query {label} skipped URL: {exc}")
                    continue
                if url not in seen:
                    seen.add(url)
                    found.append((url, hit))
                if len(found) >= self.config.max_pages:
                    break

        for index, query in enumerate(rewritten, start=1):
            collect(query, str(index))
            if len(found) >= self.config.max_pages:
                break
        if not found and question.strip().casefold() not in {
            query.strip().casefold() for query in attempted_queries
        }:
            warnings.append("Trying the original question after rewritten queries found no eligible URLs")
            collect(question.strip(), "original")
        if not found:
            warnings.append("No eligible DuckDuckGo results")

        splitter = KnowledgeRefiner(self.evaluator, self.refinement)
        candidates: list[tuple[KnowledgeStrip, RerankedHit]] = []
        source_urls: list[str] = []
        seen_pages: set[str] = set()
        for url, hit in found:
            try:
                page = self.fetcher.fetch(url)
                page_url = normalize_public_url(page.url, self.config.allowed_domains)
            except WebPageError as exc:
                warnings.append(f"Skipped {urlsplit(url).hostname}: {exc}")
                continue
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)
            text = page.text[: self.config.max_page_chars]
            spans = splitter.split_spans(text)
            if not spans:
                warnings.append(f"Skipped {urlsplit(url).hostname}: no usable text")
                continue
            if len(spans) > self.config.max_candidate_passages:
                terms = set(re.findall(r"\w{2,}", question.casefold()))
                first = set(range(min(8, self.config.max_candidate_passages, len(spans))))
                ranked = sorted(
                    range(len(spans)),
                    key=lambda index: (
                        len(terms & set(re.findall(
                            r"\w{2,}", text[spans[index][0]:spans[index][1]].casefold()
                        ))),
                        -index,
                    ),
                    reverse=True,
                )
                chosen = first | set(ranked[: self.config.max_candidate_passages - len(first)])
                spans = [spans[index] for index in sorted(chosen)]
            passages = [text[start:end] for start, end in spans]
            scores = self.reranker.score(question, passages)
            if len(scores) != len(spans) or not all(math.isfinite(score) for score in scores):
                raise ValueError("Web reranker returned inconsistent scores")
            top = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            source_urls.append(page_url)
            for index in top[: self.config.passages_per_page]:
                start, end = spans[index]
                score = scores[index]
                digest = hashlib.sha256(
                    f"{page_url}\0{page.content_sha256}\0{start}\0{end}".encode()
                ).hexdigest()[:24]
                strip_id = f"web:{digest}"
                metadata = {
                    "url": page_url,
                    "title": hit.title,
                    "search_query": hit.query,
                    "search_rank": hit.rank,
                    "page_char_start": start,
                    "page_char_end": end,
                    "content_sha256": page.content_sha256,
                    "fetched_at": page.fetched_at,
                    "page_truncated": page.truncated or len(page.text) > self.config.max_page_chars,
                    "rerank_score": score,
                }
                strip = KnowledgeStrip(
                    strip_id, passages[index], "web", page_url, metadata
                )
                chunk = Chunk(
                    strip_id, digest, index, strip.text, len(strip.text),
                    metadata={"source_url": page_url},
                )
                candidate = RerankedHit(
                    SearchResult(1.0 / (hit.rank + 1), chunk, page_url, page_url), score
                )
                candidates.append((strip, candidate))

        relevant: list[KnowledgeStrip] = []
        size = self.refinement.evaluator_batch_size
        for offset in range(0, len(candidates), size):
            batch = candidates[offset : offset + size]
            judged = self.evaluator.evaluate(question, [item[1] for item in batch])
            labels = {item.hit.result.chunk.chunk_id: item for item in judged.hits}
            expected = {item[0].strip_id for item in batch}
            if len(labels) != len(batch) or set(labels) != expected:
                raise ValueError("Web evaluator returned incomplete or duplicate judgments")
            for strip, _ in batch:
                label = labels[strip.strip_id]
                if label.relevance == Relevance.RELEVANT:
                    relevant.append(KnowledgeStrip(
                        strip.strip_id, strip.text, strip.source_type, strip.source_ref,
                        {**strip.metadata, "strip_relevance": label.relevance.value,
                         "strip_reason": label.reason},
                    ))
        relevant.sort(key=lambda strip: strip.metadata["rerank_score"], reverse=True)
        if not relevant and source_urls:
            warnings.append("No web passages passed relevance filtering")
        return WebKnowledgeResult(
            decision.action, tuple(attempted_queries), tuple(source_urls),
            tuple(relevant[: self.config.max_relevant_strips]), tuple(warnings),
        )
