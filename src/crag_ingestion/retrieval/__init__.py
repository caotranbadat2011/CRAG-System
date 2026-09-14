"""Retrieval evaluation and cited evidence preparation for CRAG."""

from .answer import AnswerGenerator, GeneratedAnswer, GeminiAnswerGenerator
from .context import AssembledContext, Citation, ContextAssembler
from .diversity import KnowledgeStrip, SemanticDiversityFilter
from .evaluator import (
    CorrectiveAction, EvaluatedHit, EvaluationDecision, GeminiRetrievalEvaluator,
    Relevance, RetrievalEvaluator,
)
from .reranker import BGEReranker, Reranker
from .refinement import KnowledgeRefiner
from .query_rewrite import GeminiQueryRewriter, QueryRewriter
from .service import RerankedHit, RerankedRetriever
from .web_fetch import FetchedPage, SafePageFetcher, WebPageError
from .web_search import (
    DDGSWebSearchProvider, PageFetcher, WebKnowledgeResult, WebKnowledgeSearcher,
    WebSearchHit, WebSearchProvider,
)

__all__ = [
    "AnswerGenerator",
    "AssembledContext",
    "BGEReranker",
    "CorrectiveAction",
    "Citation",
    "ContextAssembler",
    "EvaluatedHit",
    "EvaluationDecision",
    "FetchedPage",
    "GeneratedAnswer",
    "GeminiAnswerGenerator",
    "GeminiQueryRewriter",
    "KnowledgeStrip",
    "KnowledgeRefiner",
    "PageFetcher",
    "QueryRewriter",
    "GeminiRetrievalEvaluator",
    "Relevance",
    "RerankedHit",
    "RerankedRetriever",
    "Reranker",
    "RetrievalEvaluator",
    "SemanticDiversityFilter",
    "DDGSWebSearchProvider",
    "SafePageFetcher",
    "WebKnowledgeResult",
    "WebKnowledgeSearcher",
    "WebPageError",
    "WebSearchHit",
    "WebSearchProvider",
]
