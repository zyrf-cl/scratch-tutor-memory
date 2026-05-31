"""SQLite-backed persistence for the context-management feature.

Self-contained on purpose: it owns its own table (``context_message``)
and its own datetime helpers, and imports nothing from the parent
``store.py`` / ``service.py`` / ``schemas.py``. The only shared types are
this feature's own ``context.schemas``.

One row per message, scoped by ``(student_id, session_id)`` — the same
per-student sharding the rest of the module uses, with ``session_id``
identifying a single conversation thread. ``seq`` is a per-thread
monotonic counter (mirrors the ``dialog_summary.turn_id`` idiom).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from .schemas import (
    ContextMessage,
    ContextMessageRecord,
    ContextStats,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _schema_sql(table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    name          TEXT,
    token_count   INTEGER NOT NULL DEFAULT 0,
    pinned        INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{{}}',
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_{table}_thread
    ON {table}(student_id, session_id, seq);
"""


# Default-table schema, kept as a module constant for back-compat / callers
# that introspect it. Other tables are created via :func:`_schema_sql`.
CONTEXT_SCHEMA_SQL = _schema_sql("context_message")


class ContextStore(Protocol):
    def append_messages(
        self,
        student_id: str,
        session_id: str,
        messages: list[ContextMessage],
    ) -> list[ContextMessageRecord]: ...

    def list_messages(
        self,
        student_id: str,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> list[ContextMessageRecord]: ...

    def list_pinned(
        self, student_id: str, session_id: str
    ) -> list[ContextMessageRecord]: ...

    def stats(self, student_id: str, session_id: str) -> ContextStats: ...

    def clear(self, student_id: str, session_id: str) -> int: ...


class SQLiteContextStore:
    """Default :class:`ContextStore`. Single connection guarded by a
    lock — same shape as the parent ``SQLiteStore``."""

    def __init__(self, db_path: str = ":memory:", *, table: str = "context_message") -> None:
        # table is a trusted internal identifier (never user input); validate
        # anyway so it is always safe to interpolate into the SQL below.
        if not table.isidentifier():
            raise ValueError(f"invalid table name: {table!r}")
        self._db_path = db_path
        self._table = table
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        # By default this store shares the DB file with the parent
        # MemoryStore (a second connection over disjoint tables). Under
        # SQLite's default rollback journal a writer briefly locks the
        # whole file, so a busy timeout lets the other connection wait
        # instead of immediately raising SQLITE_BUSY. Adequate for the
        # documented low-concurrency demo workload; split onto a separate
        # file (MEMORY_MODULE_CONTEXT_DB) if write contention grows.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_schema_sql(self._table))

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_record(row: tuple) -> ContextMessageRecord:
        # row: id, seq, role, content, name, token_count, pinned,
        #      metadata_json, created_at
        return ContextMessageRecord(
            id=row[0],
            seq=row[1],
            role=row[2],
            content=row[3],
            name=row[4],
            token_count=row[5],
            pinned=bool(row[6]),
            metadata=json.loads(row[7]) if row[7] else {},
            created_at=_parse_iso(row[8]),
        )

    _SELECT_COLS = (
        "id, seq, role, content, name, token_count, pinned, "
        "metadata_json, created_at"
    )

    def append_messages(
        self,
        student_id: str,
        session_id: str,
        messages: list[ContextMessage],
    ) -> list[ContextMessageRecord]:
        if not messages:
            return []
        now_dt = _utc_now()
        now = _iso(now_dt)
        records: list[ContextMessageRecord] = []
        with self._lock:
            cur = self._conn.execute(
                f"SELECT COALESCE(MAX(seq), 0) FROM {self._table} "
                "WHERE student_id = ? AND session_id = ?",
                (student_id, session_id),
            )
            seq = int(cur.fetchone()[0])
            for msg in messages:
                seq += 1
                tokens = (
                    msg.token_count
                    if msg.token_count is not None
                    else estimate_tokens(msg.content)
                )
                meta_json = json.dumps(msg.metadata, ensure_ascii=False)
                ins = self._conn.execute(
                    f"""
                    INSERT INTO {self._table}
                        (student_id, session_id, seq, role, content, name,
                         token_count, pinned, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        student_id,
                        session_id,
                        seq,
                        msg.role,
                        msg.content,
                        msg.name,
                        tokens,
                        int(msg.pinned),
                        meta_json,
                        now,
                    ),
                )
                records.append(
                    ContextMessageRecord(
                        id=int(ins.lastrowid),
                        seq=seq,
                        role=msg.role,
                        content=msg.content,
                        name=msg.name,
                        token_count=tokens,
                        pinned=msg.pinned,
                        metadata=msg.metadata,
                        created_at=now_dt,
                    )
                )
        return records

    def list_messages(
        self,
        student_id: str,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> list[ContextMessageRecord]:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {self._SELECT_COLS} FROM {self._table} "
                "WHERE student_id = ? AND session_id = ? ORDER BY seq",
                (student_id, session_id),
            )
            rows = cur.fetchall()
        records = [self._row_to_record(r) for r in rows]
        if limit is not None:
            if limit <= 0:
                return []
            # Most recent ``limit`` messages, still in chronological order.
            records = records[-limit:]
        return records

    def list_pinned(
        self, student_id: str, session_id: str
    ) -> list[ContextMessageRecord]:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {self._SELECT_COLS} FROM {self._table} "
                "WHERE student_id = ? AND session_id = ? AND pinned = 1 "
                "ORDER BY seq",
                (student_id, session_id),
            )
            rows = cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    def stats(self, student_id: str, session_id: str) -> ContextStats:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(token_count), 0), MAX(created_at) "
                f"FROM {self._table} WHERE student_id = ? AND session_id = ?",
                (student_id, session_id),
            )
            count, tokens, last = cur.fetchone()
        return ContextStats(
            student_id=student_id,
            session_id=session_id,
            message_count=int(count),
            total_tokens=int(tokens),
            last_updated=_parse_iso(last) if last else None,
        )

    def clear(self, student_id: str, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM {self._table} "
                "WHERE student_id = ? AND session_id = ?",
                (student_id, session_id),
            )
            return int(cur.rowcount)


__all__ = ["ContextStore", "SQLiteContextStore", "CONTEXT_SCHEMA_SQL"]
