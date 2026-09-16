"""Durable local chat sessions built from completed CRAG runs."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    """SQLite history; each successful turn is committed as one user/assistant pair."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    title_custom INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    text TEXT NOT NULL,
                    run_id TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_messages_session_order
                    ON chat_messages(session_id, message_id);
                CREATE INDEX IF NOT EXISTS chat_sessions_recency
                    ON chat_sessions(updated_at DESC);
            """)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)")}
            if "title_custom" not in columns:
                conn.execute(
                    "ALTER TABLE chat_sessions ADD COLUMN title_custom INTEGER NOT NULL DEFAULT 0"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _check_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            raise ValueError("Invalid chat session ID")

    def create(self) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        stamp = _now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO chat_sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, "Cuộc trò chuyện mới", stamp, stamp),
            )
        return self.get(session_id)

    def rename(self, session_id: str, title: object) -> dict[str, Any]:
        self._check_id(session_id)
        if not isinstance(title, str):
            raise ValueError("Chat title must be text")
        normalized = " ".join(title.split())
        if not 1 <= len(normalized) <= 100:
            raise ValueError("Chat title must contain 1–100 characters")
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE chat_sessions SET title = ?, title_custom = 1, updated_at = ? WHERE session_id = ?",
                (normalized, _now(), session_id),
            )
            if cursor.rowcount == 0:
                raise KeyError("Chat session was not found")
        return self.get(session_id)

    def list(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute("""
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS turn_count
                  FROM chat_sessions AS s
                  LEFT JOIN chat_messages AS m ON m.session_id = s.session_id
                 GROUP BY s.session_id
                 ORDER BY s.updated_at DESC, s.session_id DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def get(self, session_id: str) -> dict[str, Any]:
        self._check_id(session_id)
        with closing(self._connect()) as conn:
            row = conn.execute("""
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS turn_count
                  FROM chat_sessions AS s
                  LEFT JOIN chat_messages AS m ON m.session_id = s.session_id
                 WHERE s.session_id = ? GROUP BY s.session_id
            """, (session_id,)).fetchone()
            if row is None:
                raise KeyError("Chat session was not found")
            messages = conn.execute("""
                SELECT message_id, role, text, run_id, result_json, created_at
                  FROM chat_messages WHERE session_id = ? ORDER BY message_id
            """, (session_id,)).fetchall()
        return {
            **dict(row),
            "messages": [
                {
                    "message_id": item["message_id"],
                    "role": item["role"],
                    "text": item["text"],
                    "run_id": item["run_id"],
                    "result": json.loads(item["result_json"]) if item["result_json"] else None,
                    "created_at": item["created_at"],
                }
                for item in messages
            ],
        }

    def append_exchange(
        self, session_id: str, question: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        self._check_id(session_id)
        answer = result.get("answer")
        run_id = result.get("run_id")
        if (
            not isinstance(question, str) or not question.strip()
            or (answer is not None and (
                not isinstance(answer, dict) or not isinstance(answer.get("text"), str)
            ))
            or not isinstance(run_id, str) or not SESSION_ID.fullmatch(run_id)
        ):
            raise ValueError("Chat exchange needs a question and a completed answer run")
        stamp = _now()
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT title_custom FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Chat session was not found")
            first_turn = conn.execute(
                "SELECT NOT EXISTS(SELECT 1 FROM chat_messages WHERE session_id = ?)",
                (session_id,),
            ).fetchone()[0]
            if first_turn and not row["title_custom"]:
                title = " ".join(question.split())[:72]
                conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, stamp, session_id),
                )
            else:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (stamp, session_id),
                )
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, text, created_at) VALUES (?, 'user', ?, ?)",
                (session_id, question.strip(), stamp),
            )
            conn.execute("""
                INSERT INTO chat_messages (session_id, role, text, run_id, result_json, created_at)
                VALUES (?, 'assistant', ?, ?, ?, ?)
            """, (
                session_id, answer["text"] if answer else "Lượt chạy chưa tạo được câu trả lời.", run_id,
                json.dumps(result, ensure_ascii=False), stamp,
            ))
        return self.get(session_id)

    def delete(self, session_id: str) -> None:
        self._check_id(session_id)
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
            )
            if cursor.rowcount == 0:
                raise KeyError("Chat session was not found")
