import json
from pathlib import Path

from crag_ingestion.cli import _parser, main
from crag_ingestion.embeddings.base import Embedder


def test_cli_defaults_to_bge_m3_and_its_collection() -> None:
    args = _parser().parse_args(["list"])
    assert args.embedding_model == "BAAI/bge-m3"
    assert args.qdrant_collection == "crag_bge_m3_hybrid"


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

    assert main([*common, "list"]) == 0
    list_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert len(list_output) == 1

    assert main([*common, "validate"]) == 0
    validation_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validation_output["valid"] is True
