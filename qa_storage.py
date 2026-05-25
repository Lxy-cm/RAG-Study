import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    return str(Path(os.getenv("TEMP", ".")) / "RAG-Study-main" / "qa_messages.sqlite3")


DB_PATH = os.getenv("QA_DB_PATH") or _default_db_path()
_LOCK = threading.Lock()


def _utc_now_without_timezone() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _LOCK:
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    feedback_rating TEXT,
                    feedback_comment TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS citations (
                    citation_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT,
                    snippet TEXT,
                    score REAL NOT NULL DEFAULT 1.0,
                    position INTEGER NOT NULL,
                    FOREIGN KEY (message_id)
                        REFERENCES messages(message_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON messages(conversation_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_citations_message_position
                    ON citations(message_id, position);
                """
            )


def _ensure_conversation(conn: sqlite3.Connection, conversation_id: str, title: Optional[str] = None) -> None:
    now = _utc_now_without_timezone()
    conn.execute(
        """
        INSERT INTO conversations (conversation_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET updated_at = excluded.updated_at
        """,
        (conversation_id, title, now, now),
    )


def _row_to_message(row: sqlite3.Row, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "message_id": row["message_id"],
        "role": row["role"],
        "content": row["content"],
        "citations": citations,
        "feedback_rating": row["feedback_rating"],
        "feedback_comment": row["feedback_comment"],
        "created_at": row["created_at"],
    }


def _row_to_conversation(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "conversation_id": row["conversation_id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": row["message_count"],
    }


def _load_citations(conn: sqlite3.Connection, message_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not message_ids:
        return {}

    placeholders = ",".join("?" for _ in message_ids)
    rows = conn.execute(
        f"""
        SELECT message_id, citation_id, source_type, source_id, title, snippet, score
        FROM citations
        WHERE message_id IN ({placeholders})
        ORDER BY message_id, position
        """,
        message_ids,
    ).fetchall()

    grouped: Dict[str, List[Dict[str, Any]]] = {message_id: [] for message_id in message_ids}
    for row in rows:
        grouped.setdefault(row["message_id"], []).append(
            {
                "citation_id": row["citation_id"],
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "title": row["title"],
                "snippet": row["snippet"],
                "score": row["score"],
            }
        )
    return grouped


def create_message(
    conversation_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    init_db()
    now = _utc_now_without_timezone()
    message = {
        "message_id": f"msg_{role}_{uuid.uuid4().hex[:12]}",
        "role": role,
        "content": content,
        "citations": citations or [],
        "feedback_rating": None,
        "feedback_comment": None,
        "created_at": now,
    }

    with _LOCK:
        with _connect() as conn:
            _ensure_conversation(conn, conversation_id)
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, role, content,
                    feedback_rating, feedback_comment, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message["message_id"],
                    conversation_id,
                    role,
                    content,
                    message["feedback_rating"],
                    message["feedback_comment"],
                    now,
                ),
            )
            for position, citation in enumerate(message["citations"]):
                conn.execute(
                    """
                    INSERT INTO citations (
                        citation_id, message_id, source_type, source_id,
                        title, snippet, score, position
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        citation.get("citation_id") or f"citation_{uuid.uuid4().hex[:12]}",
                        message["message_id"],
                        citation.get("source_type") or "course_content",
                        citation.get("source_id") or "",
                        citation.get("title"),
                        citation.get("snippet"),
                        float(citation.get("score") or 1.0),
                        position,
                    ),
                )

    return message


def create_conversation(title: Optional[str] = None) -> Dict[str, Any]:
    init_db()
    conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    title = title.strip() if title else None
    now = _utc_now_without_timezone()

    with _LOCK:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, title, now, now),
            )

    return {
        "conversation_id": conversation_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _LOCK:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.conversation_id
                WHERE c.conversation_id = ?
                GROUP BY c.conversation_id
                """,
                (conversation_id,),
            ).fetchone()

    return _row_to_conversation(row) if row else None


def list_conversations(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    init_db()
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    with _LOCK:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC, c.rowid DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

    return [_row_to_conversation(row) for row in rows]


def list_messages(conversation_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    init_db()
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    with _LOCK:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT message_id, role, content, feedback_rating, feedback_comment, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                LIMIT ? OFFSET ?
                """,
                (conversation_id, limit, offset),
            ).fetchall()
            message_ids = [row["message_id"] for row in rows]
            citations_by_message = _load_citations(conn, message_ids)

    return [
        _row_to_message(row, citations_by_message.get(row["message_id"], []))
        for row in rows
    ]


def migrate_json_store(json_path: Optional[str] = None) -> int:
    source_path = Path(json_path or os.getenv("QA_MESSAGE_STORE", "./data/qa_messages.json"))
    if not source_path.exists():
        return 0

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 0

    init_db()
    migrated = 0
    with _LOCK:
        with _connect() as conn:
            for conversation_id, messages in payload.items():
                if not isinstance(messages, list):
                    continue
                _ensure_conversation(conn, conversation_id)
                for message in messages:
                    if not isinstance(message, dict) or not message.get("message_id"):
                        continue
                    exists = conn.execute(
                        "SELECT 1 FROM messages WHERE message_id = ?",
                        (message["message_id"],),
                    ).fetchone()
                    if exists:
                        continue

                    conn.execute(
                        """
                        INSERT INTO messages (
                            message_id, conversation_id, role, content,
                            feedback_rating, feedback_comment, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message["message_id"],
                            conversation_id,
                            message.get("role", "assistant"),
                            message.get("content", ""),
                            message.get("feedback_rating"),
                            message.get("feedback_comment"),
                            message.get("created_at") or _utc_now_without_timezone(),
                        ),
                    )
                    for position, citation in enumerate(message.get("citations") or []):
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO citations (
                                citation_id, message_id, source_type, source_id,
                                title, snippet, score, position
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                citation.get("citation_id") or f"citation_{uuid.uuid4().hex[:12]}",
                                message["message_id"],
                                citation.get("source_type") or "course_content",
                                citation.get("source_id") or "",
                                citation.get("title"),
                                citation.get("snippet"),
                                float(citation.get("score") or 1.0),
                                position,
                            ),
                        )
                    migrated += 1
    return migrated
