"""SQLite-backed memory store implementing :class:`MemoryStore`.

One DB file, one table per memory type. Embeddings are JSON-encoded
floats stored in TEXT columns; vector search is a simple Python-side
cosine scan over the rows for one student.

This is plenty for the demo (per-student row counts in the low
hundreds). The scale-up path is to swap this implementation for one
backed by Postgres + pgvector (or a dedicated vector DB) — the
:class:`MemoryStore` protocol stays the same.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from . import schemas as S
from .embedding import cosine

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _embedding_from_json(emb_json: str) -> np.ndarray:
    return np.asarray(json.loads(emb_json), dtype=np.float32)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mastery (
    student_id     TEXT NOT NULL,
    concept_id     TEXT NOT NULL,
    score          REAL NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    last_updated   TEXT NOT NULL,
    PRIMARY KEY (student_id, concept_id)
);

CREATE TABLE IF NOT EXISTS session_state (
    student_id         TEXT PRIMARY KEY,
    last_session_at    TEXT,
    current_project_id TEXT,
    current_topic      TEXT,
    pending_task       TEXT
);

CREATE TABLE IF NOT EXISTS project (
    student_id      TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    title           TEXT NOT NULL,
    structure_json  TEXT NOT NULL,
    highlights_json TEXT NOT NULL,
    issues_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (student_id, project_id)
);

CREATE TABLE IF NOT EXISTS preference (
    student_id  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (student_id, key)
);

CREATE TABLE IF NOT EXISTS dialog_summary (
    student_id     TEXT NOT NULL,
    turn_id        INTEGER NOT NULL,
    summary        TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    embedder_dim   INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (student_id, turn_id)
);

CREATE TABLE IF NOT EXISTS error_event (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL,
    concept_id  TEXT,
    description TEXT NOT NULL,
    context     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS misconception (
    student_id     TEXT NOT NULL,
    pattern_id     TEXT NOT NULL,
    description    TEXT NOT NULL,
    occurrences    INTEGER NOT NULL DEFAULT 1,
    last_seen      TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    embedder_dim   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (student_id, pattern_id)
);

CREATE TABLE IF NOT EXISTS agent_turn (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id             TEXT NOT NULL,
    session_id             TEXT NOT NULL DEFAULT '',
    task_id                TEXT NOT NULL DEFAULT '',
    attempt_id             TEXT,
    project_name           TEXT NOT NULL DEFAULT '',
    passed                 INTEGER NOT NULL DEFAULT 0,
    score                  REAL NOT NULL DEFAULT 0.0,
    primary_error_type     TEXT NOT NULL DEFAULT '',
    concept_id             TEXT,
    feedback_level         INTEGER NOT NULL DEFAULT 1,
    feedback_text          TEXT NOT NULL DEFAULT '',
    user_text              TEXT NOT NULL DEFAULT '',
    diff_summary           TEXT NOT NULL DEFAULT '',
    highlight_node_ids_json TEXT NOT NULL DEFAULT '[]',
    need_rag               INTEGER NOT NULL DEFAULT 0,
    need_repair_validation INTEGER NOT NULL DEFAULT 0,
    created_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_turn_student_task
    ON agent_turn(student_id, task_id, id);

CREATE INDEX IF NOT EXISTS idx_agent_turn_student_session
    ON agent_turn(student_id, session_id, id);

CREATE TABLE IF NOT EXISTS affective_event (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id         TEXT NOT NULL,
    session_id         TEXT NOT NULL DEFAULT '',
    task_id            TEXT NOT NULL DEFAULT '',
    source             TEXT NOT NULL DEFAULT 'agent_inferred',
    frustration_score  REAL NOT NULL DEFAULT 0.0,
    confidence_score   REAL NOT NULL DEFAULT 0.5,
    engagement_score   REAL NOT NULL DEFAULT 0.5,
    confusion_score    REAL NOT NULL DEFAULT 0.0,
    evidence           TEXT NOT NULL DEFAULT '',
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_affective_event_student_task
    ON affective_event(student_id, task_id, id);
"""


class MemoryStore(Protocol):
    # mastery (A)
    def upsert_mastery(
        self, student_id: str, concept_id: str, score: float, note: Optional[str]
    ) -> None: ...
    def get_mastery(self, student_id: str) -> list[S.MasteryRecord]: ...

    # session (B)
    def update_session(
        self,
        student_id: str,
        last_session_at: datetime,
        current_project_id: Optional[str],
        current_topic: Optional[str],
        pending_task: Optional[str],
    ) -> None: ...
    def get_session(self, student_id: str) -> S.SessionState: ...

    # dialog (C)
    def append_dialog_summary(
        self, student_id: str, summary: str, embedding: list[float]
    ) -> int: ...
    def list_dialog_summaries(self, student_id: str) -> list[S.DialogSummaryRecord]: ...
    def search_dialog(
        self, student_id: str, query_emb: list[float], top_k: int
    ) -> list[tuple[S.DialogSummaryRecord, float]]: ...

    # projects (D)
    def save_project(self, student_id: str, evt: S.ProjectSavedEvent) -> None: ...
    def list_projects(self, student_id: str) -> list[S.ProjectRecord]: ...

    # preferences (E)
    def set_preference(self, student_id: str, evt: S.PreferenceSetEvent) -> None: ...
    def list_preferences(self, student_id: str) -> list[S.PreferenceRecord]: ...

    # misconceptions (F)
    def append_error(self, student_id: str, evt: S.ErrorObservedEvent) -> None: ...
    def list_errors(self, student_id: str) -> list[dict[str, Any]]: ...
    def upsert_misconception(
        self,
        student_id: str,
        pattern_id: str,
        description: str,
        embedding: list[float],
        *,
        occurrences: Optional[int] = None,
        last_seen: Optional[datetime] = None,
    ) -> None: ...
    def list_misconceptions(self, student_id: str) -> list[S.MisconceptionRecord]: ...
    def search_misconceptions(
        self, student_id: str, query_emb: list[float], top_k: int
    ) -> list[tuple[S.MisconceptionRecord, float]]: ...

    # agent orchestration memory
    def append_agent_turn(self, student_id: str, turn: S.AgentTurnWrite) -> int: ...
    def list_agent_turns(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[S.AgentTurnRecord]: ...
    def append_affective_event(
        self, student_id: str, event: S.AffectiveEventWrite
    ) -> int: ...
    def list_affective_events(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[S.AffectiveEventRecord]: ...


class SQLiteStore:
    """Default :class:`MemoryStore` implementation. Single connection,
    guarded by a lock — fine for the demo workload."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            # Additive column migrations for DBs created by older versions.
            # CREATE TABLE IF NOT EXISTS leaves pre-existing tables untouched,
            # so columns added after a table's first creation must be backfilled
            # here with the same DEFAULT used in SCHEMA_SQL.
            self._ensure_column(
                "agent_turn", "user_text", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                "dialog_summary", "embedder_dim", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                "misconception", "embedder_dim", "INTEGER NOT NULL DEFAULT 0"
            )

    def _ensure_column(self, table: str, column: str, column_def: str) -> None:
        """Add ``column`` to ``table`` if a legacy DB is missing it.

        ``table`` and ``column`` are internal constants (never user input),
        so interpolating them into the DDL is safe here.
        """
        cols = {
            row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in cols:
            self._conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {column_def}"
            )
            logger.info("migrated %s: added column %s", table, column)

    def close(self) -> None:
        self._conn.close()

    # ---------- mastery (A) ----------

    def upsert_mastery(
        self,
        student_id: str,
        concept_id: str,
        score: float,
        note: Optional[str] = None,
    ) -> None:
        del note  # not stored individually; evidence count is what we keep
        now = _iso(_utc_now())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO mastery (student_id, concept_id, score, evidence_count, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(student_id, concept_id) DO UPDATE SET
                    score = excluded.score,
                    evidence_count = mastery.evidence_count + 1,
                    last_updated = excluded.last_updated
                """,
                (student_id, concept_id, score, now),
            )

    def get_mastery(self, student_id: str) -> list[S.MasteryRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT concept_id, score, evidence_count, last_updated "
                "FROM mastery WHERE student_id = ? ORDER BY concept_id",
                (student_id,),
            )
            rows = cur.fetchall()
        return [
            S.MasteryRecord(
                concept_id=r[0],
                score=r[1],
                evidence_count=r[2],
                last_updated=_parse_iso(r[3]),
            )
            for r in rows
        ]

    # ---------- session (B) ----------

    def update_session(
        self,
        student_id: str,
        last_session_at: datetime,
        current_project_id: Optional[str],
        current_topic: Optional[str],
        pending_task: Optional[str],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_state
                    (student_id, last_session_at, current_project_id, current_topic, pending_task)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    last_session_at    = excluded.last_session_at,
                    current_project_id = excluded.current_project_id,
                    current_topic      = excluded.current_topic,
                    pending_task       = excluded.pending_task
                """,
                (
                    student_id,
                    _iso(last_session_at),
                    current_project_id,
                    current_topic,
                    pending_task,
                ),
            )

    def get_session(self, student_id: str) -> S.SessionState:
        with self._lock:
            cur = self._conn.execute(
                "SELECT last_session_at, current_project_id, current_topic, pending_task "
                "FROM session_state WHERE student_id = ?",
                (student_id,),
            )
            row = cur.fetchone()
        if row is None:
            return S.SessionState()
        return S.SessionState(
            last_session_at=_parse_iso(row[0]) if row[0] else None,
            current_project_id=row[1],
            current_topic=row[2],
            pending_task=row[3],
        )

    # ---------- dialog (C) ----------

    def append_dialog_summary(
        self, student_id: str, summary: str, embedding: list[float]
    ) -> int:
        now = _iso(_utc_now())
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(turn_id), 0) FROM dialog_summary WHERE student_id = ?",
                (student_id,),
            )
            next_id = int(cur.fetchone()[0]) + 1
            self._conn.execute(
                "INSERT INTO dialog_summary "
                "(student_id, turn_id, summary, embedding_json, embedder_dim, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    student_id,
                    next_id,
                    summary,
                    json.dumps(embedding),
                    len(embedding),
                    now,
                ),
            )
        return next_id

    def list_dialog_summaries(self, student_id: str) -> list[S.DialogSummaryRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT turn_id, summary, created_at FROM dialog_summary "
                "WHERE student_id = ? ORDER BY turn_id",
                (student_id,),
            )
            rows = cur.fetchall()
        return [
            S.DialogSummaryRecord(turn_id=r[0], summary=r[1], created_at=_parse_iso(r[2]))
            for r in rows
        ]

    def search_dialog(
        self, student_id: str, query_emb: list[float], top_k: int
    ) -> list[tuple[S.DialogSummaryRecord, float]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT turn_id, summary, embedding_json, embedder_dim, created_at "
                "FROM dialog_summary WHERE student_id = ?",
                (student_id,),
            )
            rows = cur.fetchall()
        q = np.asarray(query_emb, dtype=np.float32)
        scored: list[tuple[S.DialogSummaryRecord, float]] = []
        for turn_id, summary, emb_json, embedder_dim, created_at in rows:
            v = _embedding_from_json(emb_json)
            if v.shape != q.shape:
                logger.debug(
                    "skip dialog embedding dim mismatch student=%s turn=%s stored=%s query=%s",
                    student_id,
                    turn_id,
                    embedder_dim or len(v),
                    len(q),
                )
                continue
            score = cosine(q, v)
            scored.append(
                (
                    S.DialogSummaryRecord(
                        turn_id=turn_id, summary=summary, created_at=_parse_iso(created_at)
                    ),
                    score,
                )
            )
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    # ---------- projects (D) ----------

    def save_project(self, student_id: str, evt: S.ProjectSavedEvent) -> None:
        now = _iso(_utc_now())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO project
                    (student_id, project_id, title, structure_json, highlights_json,
                     issues_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(student_id, project_id) DO UPDATE SET
                    title           = excluded.title,
                    structure_json  = excluded.structure_json,
                    highlights_json = excluded.highlights_json,
                    issues_json     = excluded.issues_json,
                    created_at      = excluded.created_at
                """,
                (
                    student_id,
                    evt.project_id,
                    evt.title,
                    json.dumps(evt.structure),
                    json.dumps(evt.highlights),
                    json.dumps(evt.issues),
                    now,
                ),
            )

    def list_projects(self, student_id: str) -> list[S.ProjectRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT project_id, title, structure_json, highlights_json, "
                "issues_json, created_at FROM project "
                "WHERE student_id = ? ORDER BY created_at",
                (student_id,),
            )
            rows = cur.fetchall()
        return [
            S.ProjectRecord(
                project_id=r[0],
                title=r[1],
                structure=json.loads(r[2]),
                highlights=json.loads(r[3]),
                issues=json.loads(r[4]),
                created_at=_parse_iso(r[5]),
            )
            for r in rows
        ]

    # ---------- preferences (E) ----------

    def set_preference(self, student_id: str, evt: S.PreferenceSetEvent) -> None:
        now = _iso(_utc_now())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO preference (student_id, key, value, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(student_id, key) DO UPDATE SET
                    value = excluded.value,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (student_id, evt.key, evt.value, evt.source, now),
            )

    def list_preferences(self, student_id: str) -> list[S.PreferenceRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT key, value, source, updated_at FROM preference "
                "WHERE student_id = ? ORDER BY key",
                (student_id,),
            )
            rows = cur.fetchall()
        return [
            S.PreferenceRecord(
                key=r[0], value=r[1], source=r[2], updated_at=_parse_iso(r[3])
            )
            for r in rows
        ]

    # ---------- errors + misconceptions (F) ----------

    def append_error(self, student_id: str, evt: S.ErrorObservedEvent) -> None:
        now = _iso(_utc_now())
        with self._lock:
            self._conn.execute(
                "INSERT INTO error_event "
                "(student_id, concept_id, description, context, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (student_id, evt.concept_id, evt.description, evt.context, now),
            )

    def list_errors(self, student_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, concept_id, description, context, created_at "
                "FROM error_event WHERE student_id = ? ORDER BY id",
                (student_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "concept_id": r[1],
                "description": r[2],
                "context": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def upsert_misconception(
        self,
        student_id: str,
        pattern_id: str,
        description: str,
        embedding: list[float],
        *,
        occurrences: Optional[int] = None,
        last_seen: Optional[datetime] = None,
    ) -> None:
        seen_at = _iso(last_seen) if last_seen is not None else _iso(_utc_now())
        emb_json = json.dumps(embedding)
        with self._lock:
            if occurrences is None:
                # No authoritative count supplied: preserve the legacy
                # increment-on-conflict behaviour as a fallback.
                self._conn.execute(
                    """
                    INSERT INTO misconception
                        (student_id, pattern_id, description, occurrences,
                         last_seen, embedding_json, embedder_dim)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(student_id, pattern_id) DO UPDATE SET
                        description    = excluded.description,
                        occurrences    = misconception.occurrences + 1,
                        last_seen      = excluded.last_seen,
                        embedding_json = excluded.embedding_json,
                        embedder_dim   = excluded.embedder_dim
                    """,
                    (student_id, pattern_id, description, seen_at, emb_json, len(embedding)),
                )
            else:
                # Authoritative count from the detector: set it directly so
                # occurrences reflects the true number of contributing errors
                # and last_seen tracks the most recent one.
                self._conn.execute(
                    """
                    INSERT INTO misconception
                        (student_id, pattern_id, description, occurrences,
                         last_seen, embedding_json, embedder_dim)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(student_id, pattern_id) DO UPDATE SET
                        description    = excluded.description,
                        occurrences    = excluded.occurrences,
                        last_seen      = excluded.last_seen,
                        embedding_json = excluded.embedding_json,
                        embedder_dim   = excluded.embedder_dim
                    """,
                    (
                        student_id,
                        pattern_id,
                        description,
                        int(occurrences),
                        seen_at,
                        emb_json,
                        len(embedding),
                    ),
                )

    def list_misconceptions(self, student_id: str) -> list[S.MisconceptionRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT pattern_id, description, occurrences, last_seen "
                "FROM misconception WHERE student_id = ? ORDER BY pattern_id",
                (student_id,),
            )
            rows = cur.fetchall()
        return [
            S.MisconceptionRecord(
                pattern_id=r[0],
                description=r[1],
                occurrences=r[2],
                last_seen=_parse_iso(r[3]),
            )
            for r in rows
        ]

    def search_misconceptions(
        self, student_id: str, query_emb: list[float], top_k: int
    ) -> list[tuple[S.MisconceptionRecord, float]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT pattern_id, description, occurrences, last_seen, embedding_json, embedder_dim "
                "FROM misconception WHERE student_id = ?",
                (student_id,),
            )
            rows = cur.fetchall()
        q = np.asarray(query_emb, dtype=np.float32)
        scored: list[tuple[S.MisconceptionRecord, float]] = []
        for pattern_id, description, occurrences, last_seen, emb_json, embedder_dim in rows:
            v = _embedding_from_json(emb_json)
            if v.shape != q.shape:
                logger.debug(
                    "skip misconception embedding dim mismatch student=%s pattern=%s stored=%s query=%s",
                    student_id,
                    pattern_id,
                    embedder_dim or len(v),
                    len(q),
                )
                continue
            score = cosine(q, v)
            scored.append(
                (
                    S.MisconceptionRecord(
                        pattern_id=pattern_id,
                        description=description,
                        occurrences=occurrences,
                        last_seen=_parse_iso(last_seen),
                    ),
                    score,
                )
            )
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    # ---------- agent orchestration turns ----------

    def append_agent_turn(self, student_id: str, turn: S.AgentTurnWrite) -> int:
        now = _iso(_utc_now())
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO agent_turn
                    (student_id, session_id, task_id, attempt_id, project_name,
                     passed, score, primary_error_type, concept_id,
                     feedback_level, feedback_text, user_text, diff_summary,
                     highlight_node_ids_json, need_rag,
                     need_repair_validation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    turn.session_id,
                    turn.task_id,
                    turn.attempt_id,
                    turn.project_name,
                    int(turn.passed),
                    turn.score,
                    turn.primary_error_type,
                    turn.concept_id,
                    turn.feedback_level,
                    turn.feedback_text,
                    turn.user_text,
                    turn.diff_summary,
                    json.dumps(turn.highlight_node_ids, ensure_ascii=False),
                    int(turn.need_rag),
                    int(turn.need_repair_validation),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def list_agent_turns(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[S.AgentTurnRecord]:
        where = ["student_id = ?"]
        params: list[Any] = [student_id]
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if task_id:
            where.append("task_id = ?")
            params.append(task_id)
        params.append(max(1, limit))

        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, session_id, task_id, attempt_id, project_name,
                       passed, score, primary_error_type, concept_id,
                       feedback_level, feedback_text, user_text, diff_summary,
                       highlight_node_ids_json, need_rag,
                       need_repair_validation, created_at
                FROM agent_turn
                WHERE """ + " AND ".join(where) + """
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            )
            rows = cur.fetchall()

        records = [
            S.AgentTurnRecord(
                id=r[0],
                session_id=r[1],
                task_id=r[2],
                attempt_id=r[3],
                project_name=r[4],
                passed=bool(r[5]),
                score=r[6],
                primary_error_type=r[7],
                concept_id=r[8],
                feedback_level=r[9],
                feedback_text=r[10],
                user_text=r[11],
                diff_summary=r[12],
                highlight_node_ids=json.loads(r[13]),
                need_rag=bool(r[14]),
                need_repair_validation=bool(r[15]),
                created_at=_parse_iso(r[16]),
            )
            for r in rows
        ]
        records.reverse()
        return records

    def append_affective_event(
        self,
        student_id: str,
        event: S.AffectiveEventWrite,
    ) -> int:
        now = _iso(_utc_now())
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO affective_event
                    (student_id, session_id, task_id, source,
                     frustration_score, confidence_score, engagement_score,
                     confusion_score, evidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    event.session_id,
                    event.task_id,
                    event.source,
                    event.frustration_score,
                    event.confidence_score,
                    event.engagement_score,
                    event.confusion_score,
                    event.evidence,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def list_affective_events(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[S.AffectiveEventRecord]:
        where = ["student_id = ?"]
        params: list[Any] = [student_id]
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if task_id:
            where.append("task_id = ?")
            params.append(task_id)
        params.append(max(1, limit))

        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, session_id, task_id, source, frustration_score,
                       confidence_score, engagement_score, confusion_score,
                       evidence, created_at
                FROM affective_event
                WHERE """ + " AND ".join(where) + """
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            )
            rows = cur.fetchall()

        records = [
            S.AffectiveEventRecord(
                id=r[0],
                session_id=r[1],
                task_id=r[2],
                source=r[3],
                frustration_score=r[4],
                confidence_score=r[5],
                engagement_score=r[6],
                confusion_score=r[7],
                evidence=r[8],
                created_at=_parse_iso(r[9]),
            )
            for r in rows
        ]
        records.reverse()
        return records


__all__ = ["MemoryStore", "SQLiteStore", "SCHEMA_SQL"]
