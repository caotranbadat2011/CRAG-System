from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from crag_ingestion.chat import ChatStore


def _result(run_id: str = "a" * 32) -> dict[str, object]:
    return {
        "run_id": run_id,
        "branch": "Correct",
        "answer": {"text": "Đáp án có nguồn [1].", "status": "answered"},
        "citations": [{"marker": "[1]", "text": "Nguồn gốc", "viewer_url": "/api/runs/" + run_id}],
    }


def test_chat_store_persists_turns_and_source_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "chats.sqlite3"
    store = ChatStore(path)
    created = store.create()
    session_id = created["session_id"]
    assert created["messages"] == []
    assert created["turn_count"] == 0

    first = store.append_exchange(session_id, "  Tài liệu nói gì?  ", _result())
    assert first["title"] == "Tài liệu nói gì?"
    assert first["turn_count"] == 1
    assert [item["role"] for item in first["messages"]] == ["user", "assistant"]
    assert first["messages"][0]["text"] == "Tài liệu nói gì?"
    assert first["messages"][1]["run_id"] == "a" * 32
    assert first["messages"][1]["result"]["citations"][0]["marker"] == "[1]"

    reopened = ChatStore(path)
    second = reopened.append_exchange(session_id, "Câu tiếp theo?", _result("b" * 32))
    assert second["title"] == "Tài liệu nói gì?"
    assert second["turn_count"] == 2
    assert len(second["messages"]) == 4
    assert reopened.list()[0]["session_id"] == session_id
    assert reopened.list()[0]["turn_count"] == 2

    reopened.delete(session_id)
    assert reopened.list() == []
    with pytest.raises(KeyError):
        store.get(session_id)


def test_chat_store_rejects_incomplete_turn_without_mutation(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats.sqlite3")
    session_id = store.create()["session_id"]
    with pytest.raises(ValueError):
        store.append_exchange(session_id, "Question", {"run_id": "bad", "answer": {"text": "A"}})
    with pytest.raises(ValueError):
        store.append_exchange(session_id, " ", _result())
    assert store.get(session_id)["messages"] == []
    with pytest.raises(KeyError):
        store.delete("f" * 32)


def test_chat_store_keeps_completed_run_without_answer(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats.sqlite3")
    session_id = store.create()["session_id"]
    run = _result()
    run["answer"] = None
    session = store.append_exchange(session_id, "Question", run)
    assert session["turn_count"] == 1
    assert session["messages"][1]["result"]["answer"] is None


def test_chat_rename_survives_first_question_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "chats.sqlite3"
    store = ChatStore(path)
    session_id = store.create()["session_id"]
    with pytest.raises(ValueError):
        store.rename(session_id, "  ")
    with pytest.raises(ValueError):
        store.rename(session_id, "x" * 101)
    renamed = store.rename(session_id, "  Đồ án   hỏi–đáp  ")
    assert renamed["title"] == "Đồ án hỏi–đáp"
    assert ChatStore(path).append_exchange(session_id, "First question?", _result())["title"] == "Đồ án hỏi–đáp"
    assert store.rename(session_id, "Tên mới")["title"] == "Tên mới"
    assert ChatStore(path).get(session_id)["title"] == "Tên mới"


def test_chat_store_migrates_existing_sessions(tmp_path: Path) -> None:
    path = tmp_path / "chats.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    store = ChatStore(path)
    session_id = store.create()["session_id"]
    assert store.rename(session_id, "Tên từ bản cũ")["title"] == "Tên từ bản cũ"
