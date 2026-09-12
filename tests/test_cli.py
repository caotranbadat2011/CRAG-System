import json
from pathlib import Path

from crag_ingestion.cli import main
from crag_ingestion.embeddings.base import Embedder


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

    assert main([*common, "list"]) == 0
    list_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert len(list_output) == 1

    assert main([*common, "validate"]) == 0
    validation_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validation_output["valid"] is True
