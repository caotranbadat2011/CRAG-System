import json
from pathlib import Path

from crag_ingestion.cli import _parser, main
from crag_ingestion.embeddings.base import Embedder
from crag_ingestion.retrieval import CorrectiveAction, EvaluationDecision, KnowledgeStrip


def test_cli_defaults_to_bge_m3_and_its_collection() -> None:
    args = _parser().parse_args(["list"])
    assert args.embedding_model == "BAAI/bge-m3"
    assert args.qdrant_collection == "crag_bge_m3_hybrid"


def test_cli_exposes_refinement_limits() -> None:
    args = _parser().parse_args([
        "--strip-max-chars", "240", "--strip-max-sentences", "1",
        "--strip-batch-size", "5", "refine", "question",
    ])
    assert (args.strip_max_chars, args.strip_max_sentences, args.strip_batch_size) == (240, 1, 5)


def test_cli_exposes_answer_length_limits() -> None:
    args = _parser().parse_args([
        "--answer-max-claims", "16", "--answer-max-output-tokens", "6144", "run", "question"
    ])
    assert (args.answer_max_claims, args.answer_max_output_tokens) == (16, 6144)


def test_cli_exposes_local_web_server_port() -> None:
    args = _parser().parse_args(["serve", "--port", "8123"])
    assert args.command == "serve" and args.port == 8123


def test_cli_ingest_query_list_and_validate(
    tmp_path: Path, capsys: object, monkeypatch: object, fake_embedder: Embedder
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "crag_ingestion.pipeline.create_embedder", lambda _config: fake_embedder
    )
    source = tmp_path / "knowledge.md"
    source.write_text("# CRAG\nCorrective retrieval kiểm tra chất lượng tài liệu.", encoding="utf-8")
    data_dir = tmp_path / "runtime"
    common = ["--data-dir", str(data_dir)]

    assert main([*common, "ingest", str(source)]) == 0
    ingest_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert ingest_output["discovered"] == 1
    assert ingest_output["failures"] == []

    assert main([*common, "query", "retrieval chất lượng", "--limit", "1"]) == 0
    query_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert len(query_output) == 1
    assert query_output[0]["score"] > 0

    class FakeReranker:
        name = "test/reranker"

        def score(self, _query: str, passages: list[str]) -> list[float]:
            return [float(len(passage)) for passage in passages]

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "crag_ingestion.pipeline.BGEReranker", lambda _model_name: FakeReranker()
    )
    assert main([*common, "query", "retrieval chất lượng", "--limit", "1", "--rerank"]) == 0
    reranked_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert len(reranked_output) == 1
    assert "retrieval_score" in reranked_output[0]
    assert "rerank_score" in reranked_output[0]

    class FakeEvaluator:
        def evaluate(self, _query: str, hits: list[object]) -> object:
            class Decision:
                def to_dict(self) -> dict[str, object]:
                    return {"action": "Correct", "model": "test/evaluator", "hits": []}

            assert hits
            return Decision()

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "crag_ingestion.pipeline.GeminiRetrievalEvaluator",
        lambda _model_name: FakeEvaluator(),
    )
    assert main([*common, "evaluate", "retrieval chất lượng", "--limit", "1"]) == 0
    evaluation_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert evaluation_output["action"] == "Correct"

    assert main([*common, "list"]) == 0
    list_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert len(list_output) == 1

    assert main([*common, "validate"]) == 0
    validation_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validation_output["valid"] is True


def test_cli_refine_returns_decision_and_internal_strips(
    capsys: object, monkeypatch: object,
) -> None:
    class FakePipeline:
        def __init__(self, _config: object) -> None:
            pass

        def __enter__(self) -> "FakePipeline":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def evaluate_retrieval(self, query: str, **_kwargs: object) -> EvaluationDecision:
            assert query == "hoàn tiền"
            return EvaluationDecision(CorrectiveAction.CORRECT, "fake/model", ())

        def refine_internal_knowledge(
            self, query: str, decision: EvaluationDecision
        ) -> list[KnowledgeStrip]:
            assert query == "hoàn tiền"
            assert decision.action == CorrectiveAction.CORRECT
            return [KnowledgeStrip("s1", "Hoàn tiền trong 30 ngày.", "internal", "chunk-1")]

    monkeypatch.setattr("crag_ingestion.cli.IngestionPipeline", FakePipeline)  # type: ignore[attr-defined]
    assert main(["refine", "hoàn tiền", "--limit", "3"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["decision"]["action"] == "Correct"
    assert output["internal_strips"][0]["source_ref"] == "chunk-1"
