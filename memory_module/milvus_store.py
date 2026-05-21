"""Milvus-backed MemoryStore implementation.

This store keeps the public MemoryStore contract identical to SQLiteStore,
but persists records in Milvus collections. The dialog and misconception
channels use real vectors for semantic recall; the other channels use a
stable placeholder vector and treat Milvus as a typed document store.

Milvus is strongest when row counts and vector-search traffic grow. For a
small local demo, SQLite remains simpler and is still the default backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from . import schemas as S

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _ts(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000)


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _student_filter(student_id: str) -> str:
    return f"student_id == {_json_string(student_id)}"


def _student_record_filter(student_id: str, record_key: str) -> str:
    return (
        f"student_id == {_json_string(student_id)} "
        f"and record_key == {_json_string(record_key)}"
    )


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("payload") or {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _hit_entity(hit: dict[str, Any]) -> dict[str, Any]:
    entity = hit.get("entity") or {}
    return entity if isinstance(entity, dict) else {}


class MilvusStore:
    """Milvus implementation of the MemoryStore protocol.

    Configuration can be passed explicitly or through environment variables:

    * ``MEMORY_MODULE_MILVUS_URI``: Milvus URI, default ``http://localhost:19530``
    * ``MEMORY_MODULE_MILVUS_TOKEN``: token or ``user:password``, optional
    * ``MEMORY_MODULE_MILVUS_DB``: database name, optional
    * ``MEMORY_MODULE_MILVUS_PREFIX``: collection prefix, default ``memory_module``
    * ``MEMORY_MODULE_MILVUS_TIMEOUT``: operation timeout seconds, default ``5``
    * ``MEMORY_MODULE_MILVUS_CONSISTENCY``: default ``Strong``
    """

    _COLLECTIONS = (
        "mastery",
        "session_state",
        "project",
        "preference",
        "dialog_summary",
        "error_event",
        "misconception",
        "agent_turn",
        "affective_event",
    )
    _OUTPUT_FIELDS = ["id", "student_id", "record_key", "created_at_ts", "payload"]
    _QUERY_LIMIT = 16_384

    def __init__(
        self,
        *,
        uri: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        collection_prefix: str | None = None,
        vector_dim: int = 64,
        timeout: float | None = None,
        consistency_level: str | int | None = None,
        client: Any | None = None,
    ) -> None:
        if vector_dim <= 1:
            raise ValueError("Milvus vector_dim must be greater than 1")

        self._uri = uri or os.environ.get("MEMORY_MODULE_MILVUS_URI") or "http://localhost:19530"
        self._token = token if token is not None else os.environ.get("MEMORY_MODULE_MILVUS_TOKEN")
        self._db_name = db_name if db_name is not None else os.environ.get("MEMORY_MODULE_MILVUS_DB")
        self._prefix = collection_prefix or os.environ.get("MEMORY_MODULE_MILVUS_PREFIX") or "memory_module"
        self._dim = int(vector_dim)
        self._timeout = (
            float(timeout)
            if timeout is not None
            else float(os.environ.get("MEMORY_MODULE_MILVUS_TIMEOUT", "5"))
        )
        self._consistency = (
            consistency_level
            if consistency_level is not None
            else os.environ.get("MEMORY_MODULE_MILVUS_CONSISTENCY", "Strong")
        )

        if client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "pymilvus is not installed. Install with: uv sync --extra milvus"
                ) from exc

            kwargs: dict[str, Any] = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            if self._db_name:
                kwargs["db_name"] = self._db_name
            client = MilvusClient(**kwargs)

        self._client = client
        self._ensure_collections()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()

    # ---------- collection plumbing ----------

    def _collection(self, suffix: str) -> str:
        return f"{self._prefix}_{suffix}"

    def _pk(self, suffix: str, *parts: str) -> str:
        raw = "\x1f".join((suffix, *parts))
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"{suffix}_{digest}"

    def _placeholder_vector(self) -> list[float]:
        vector = [0.0] * self._dim
        vector[0] = 1.0
        return vector

    def _checked_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self._dim:
            raise ValueError(
                f"Milvus vector dimension mismatch: got {len(vector)}, expected {self._dim}"
            )
        return [float(v) for v in vector]

    def _ensure_collections(self) -> None:
        try:
            from pymilvus import DataType
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pymilvus is not installed. Install with: uv sync --extra milvus"
            ) from exc

        for suffix in self._COLLECTIONS:
            name = self._collection(suffix)
            if self._client.has_collection(name, timeout=self._timeout):
                continue

            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=512,
            )
            schema.add_field(
                field_name="student_id",
                datatype=DataType.VARCHAR,
                max_length=256,
            )
            schema.add_field(
                field_name="record_key",
                datatype=DataType.VARCHAR,
                max_length=512,
            )
            schema.add_field(
                field_name="created_at_ts",
                datatype=DataType.INT64,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._dim,
            )
            schema.add_field(field_name="payload", datatype=DataType.JSON)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )

            self._client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
                consistency_level=self._consistency,
                timeout=self._timeout,
            )
            logger.info("created Milvus collection %s dim=%d", name, self._dim)

    def _upsert_record(
        self,
        suffix: str,
        *,
        pk: str,
        student_id: str,
        record_key: str,
        payload: dict[str, Any],
        vector: list[float] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        dt = created_at or _utc_now()
        self._client.upsert(
            collection_name=self._collection(suffix),
            data={
                "id": pk,
                "student_id": student_id,
                "record_key": record_key,
                "created_at_ts": _ts(dt),
                "vector": self._checked_vector(vector or self._placeholder_vector()),
                "payload": payload,
            },
            timeout=self._timeout,
        )

    def _query(
        self,
        suffix: str,
        expr: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return list(
            self._client.query(
                collection_name=self._collection(suffix),
                filter=expr,
                output_fields=self._OUTPUT_FIELDS,
                limit=limit or self._QUERY_LIMIT,
                timeout=self._timeout,
            )
            or []
        )

    def _query_student(self, suffix: str, student_id: str) -> list[dict[str, Any]]:
        return self._query(suffix, _student_filter(student_id))

    def _query_one(
        self,
        suffix: str,
        student_id: str,
        record_key: str,
    ) -> dict[str, Any] | None:
        rows = self._query(
            suffix,
            _student_record_filter(student_id, record_key),
            limit=1,
        )
        return rows[0] if rows else None

    def _next_payload_id(self, suffix: str, student_id: str) -> int:
        rows = self._query_student(suffix, student_id)
        ids = [int(_payload(row).get("id") or 0) for row in rows]
        return (max(ids) if ids else 0) + 1

    # ---------- mastery (A) ----------

    def upsert_mastery(
        self,
        student_id: str,
        concept_id: str,
        score: float,
        note: Optional[str] = None,
    ) -> None:
        del note
        now = _utc_now()
        existing = self._query_one("mastery", student_id, concept_id)
        evidence_count = 1
        if existing:
            evidence_count = int(_payload(existing).get("evidence_count") or 0) + 1
        self._upsert_record(
            "mastery",
            pk=self._pk("mastery", student_id, concept_id),
            student_id=student_id,
            record_key=concept_id,
            payload={
                "concept_id": concept_id,
                "score": float(score),
                "evidence_count": evidence_count,
                "last_updated": _iso(now),
            },
            created_at=now,
        )

    def get_mastery(self, student_id: str) -> list[S.MasteryRecord]:
        rows = self._query_student("mastery", student_id)
        records = [
            S.MasteryRecord(
                concept_id=p["concept_id"],
                score=float(p["score"]),
                evidence_count=int(p["evidence_count"]),
                last_updated=_parse_iso(p["last_updated"]),
            )
            for p in (_payload(row) for row in rows)
        ]
        records.sort(key=lambda rec: rec.concept_id)
        return records

    # ---------- session (B) ----------

    def update_session(
        self,
        student_id: str,
        last_session_at: datetime,
        current_project_id: Optional[str],
        current_topic: Optional[str],
        pending_task: Optional[str],
    ) -> None:
        self._upsert_record(
            "session_state",
            pk=self._pk("session_state", student_id),
            student_id=student_id,
            record_key="session",
            payload={
                "last_session_at": _iso(last_session_at),
                "current_project_id": current_project_id,
                "current_topic": current_topic,
                "pending_task": pending_task,
            },
            created_at=last_session_at,
        )

    def get_session(self, student_id: str) -> S.SessionState:
        row = self._query_one("session_state", student_id, "session")
        if not row:
            return S.SessionState()
        p = _payload(row)
        return S.SessionState(
            last_session_at=_parse_iso(p["last_session_at"]) if p.get("last_session_at") else None,
            current_project_id=p.get("current_project_id"),
            current_topic=p.get("current_topic"),
            pending_task=p.get("pending_task"),
        )

    # ---------- dialog (C) ----------

    def append_dialog_summary(
        self,
        student_id: str,
        summary: str,
        embedding: list[float],
    ) -> int:
        rows = self._query_student("dialog_summary", student_id)
        ids = [int(_payload(row).get("turn_id") or 0) for row in rows]
        turn_id = (max(ids) if ids else 0) + 1
        now = _utc_now()
        self._upsert_record(
            "dialog_summary",
            pk=self._pk("dialog_summary", student_id, str(turn_id)),
            student_id=student_id,
            record_key=f"{turn_id:020d}",
            payload={
                "turn_id": turn_id,
                "summary": summary,
                "created_at": _iso(now),
            },
            vector=embedding,
            created_at=now,
        )
        return turn_id

    def list_dialog_summaries(self, student_id: str) -> list[S.DialogSummaryRecord]:
        rows = self._query_student("dialog_summary", student_id)
        records = [
            S.DialogSummaryRecord(
                turn_id=int(p["turn_id"]),
                summary=p["summary"],
                created_at=_parse_iso(p["created_at"]),
            )
            for p in (_payload(row) for row in rows)
        ]
        records.sort(key=lambda rec: rec.turn_id)
        return records

    def search_dialog(
        self,
        student_id: str,
        query_emb: list[float],
        top_k: int,
    ) -> list[tuple[S.DialogSummaryRecord, float]]:
        return self._search_payload_records(
            "dialog_summary",
            student_id=student_id,
            query_emb=query_emb,
            top_k=top_k,
            factory=lambda p: S.DialogSummaryRecord(
                turn_id=int(p["turn_id"]),
                summary=p["summary"],
                created_at=_parse_iso(p["created_at"]),
            ),
        )

    # ---------- projects (D) ----------

    def save_project(self, student_id: str, evt: S.ProjectSavedEvent) -> None:
        now = _utc_now()
        self._upsert_record(
            "project",
            pk=self._pk("project", student_id, evt.project_id),
            student_id=student_id,
            record_key=evt.project_id,
            payload={
                "project_id": evt.project_id,
                "title": evt.title,
                "structure": evt.structure,
                "highlights": evt.highlights,
                "issues": evt.issues,
                "created_at": _iso(now),
            },
            created_at=now,
        )

    def list_projects(self, student_id: str) -> list[S.ProjectRecord]:
        rows = self._query_student("project", student_id)
        records = [
            S.ProjectRecord(
                project_id=p["project_id"],
                title=p["title"],
                structure=dict(p.get("structure") or {}),
                highlights=list(p.get("highlights") or []),
                issues=list(p.get("issues") or []),
                created_at=_parse_iso(p["created_at"]),
            )
            for p in (_payload(row) for row in rows)
        ]
        records.sort(key=lambda rec: rec.created_at)
        return records

    # ---------- preferences (E) ----------

    def set_preference(self, student_id: str, evt: S.PreferenceSetEvent) -> None:
        now = _utc_now()
        self._upsert_record(
            "preference",
            pk=self._pk("preference", student_id, evt.key),
            student_id=student_id,
            record_key=evt.key,
            payload={
                "key": evt.key,
                "value": evt.value,
                "source": evt.source,
                "updated_at": _iso(now),
            },
            created_at=now,
        )

    def list_preferences(self, student_id: str) -> list[S.PreferenceRecord]:
        rows = self._query_student("preference", student_id)
        records = [
            S.PreferenceRecord(
                key=p["key"],
                value=p["value"],
                source=p["source"],
                updated_at=_parse_iso(p["updated_at"]),
            )
            for p in (_payload(row) for row in rows)
        ]
        records.sort(key=lambda rec: rec.key)
        return records

    # ---------- errors + misconceptions (F) ----------

    def append_error(self, student_id: str, evt: S.ErrorObservedEvent) -> None:
        next_id = self._next_payload_id("error_event", student_id)
        now = _utc_now()
        self._upsert_record(
            "error_event",
            pk=self._pk("error_event", student_id, str(next_id)),
            student_id=student_id,
            record_key=f"{next_id:020d}",
            payload={
                "id": next_id,
                "concept_id": evt.concept_id,
                "description": evt.description,
                "context": evt.context,
                "created_at": _iso(now),
            },
            created_at=now,
        )

    def list_errors(self, student_id: str) -> list[dict[str, Any]]:
        rows = self._query_student("error_event", student_id)
        errors = [_payload(row) for row in rows]
        errors.sort(key=lambda item: int(item.get("id") or 0))
        return errors

    def upsert_misconception(
        self,
        student_id: str,
        pattern_id: str,
        description: str,
        embedding: list[float],
    ) -> None:
        now = _utc_now()
        existing = self._query_one("misconception", student_id, pattern_id)
        occurrences = 1
        if existing:
            occurrences = int(_payload(existing).get("occurrences") or 0) + 1
        self._upsert_record(
            "misconception",
            pk=self._pk("misconception", student_id, pattern_id),
            student_id=student_id,
            record_key=pattern_id,
            payload={
                "pattern_id": pattern_id,
                "description": description,
                "occurrences": occurrences,
                "last_seen": _iso(now),
            },
            vector=embedding,
            created_at=now,
        )

    def list_misconceptions(self, student_id: str) -> list[S.MisconceptionRecord]:
        rows = self._query_student("misconception", student_id)
        records = [
            S.MisconceptionRecord(
                pattern_id=p["pattern_id"],
                description=p["description"],
                occurrences=int(p["occurrences"]),
                last_seen=_parse_iso(p["last_seen"]),
            )
            for p in (_payload(row) for row in rows)
        ]
        records.sort(key=lambda rec: rec.pattern_id)
        return records

    def search_misconceptions(
        self,
        student_id: str,
        query_emb: list[float],
        top_k: int,
    ) -> list[tuple[S.MisconceptionRecord, float]]:
        return self._search_payload_records(
            "misconception",
            student_id=student_id,
            query_emb=query_emb,
            top_k=top_k,
            factory=lambda p: S.MisconceptionRecord(
                pattern_id=p["pattern_id"],
                description=p["description"],
                occurrences=int(p["occurrences"]),
                last_seen=_parse_iso(p["last_seen"]),
            ),
        )

    def _search_payload_records(
        self,
        suffix: str,
        *,
        student_id: str,
        query_emb: list[float],
        top_k: int,
        factory: Any,
    ) -> list[tuple[Any, float]]:
        result = self._client.search(
            collection_name=self._collection(suffix),
            data=[self._checked_vector(query_emb)],
            filter=_student_filter(student_id),
            limit=max(1, int(top_k)),
            output_fields=["payload"],
            search_params={"metric_type": "COSINE", "params": {}},
            timeout=self._timeout,
        )
        hits = result[0] if result else []
        records: list[tuple[Any, float]] = []
        for hit in hits:
            p = _payload(_hit_entity(hit))
            if not p:
                continue
            records.append((factory(p), float(hit.get("distance", 0.0))))
        return records

    # ---------- agent orchestration turns ----------

    def append_agent_turn(self, student_id: str, turn: S.AgentTurnWrite) -> int:
        next_id = self._next_payload_id("agent_turn", student_id)
        now = _utc_now()
        self._upsert_record(
            "agent_turn",
            pk=self._pk("agent_turn", student_id, str(next_id)),
            student_id=student_id,
            record_key=f"{next_id:020d}",
            payload={
                "id": next_id,
                "session_id": turn.session_id,
                "task_id": turn.task_id,
                "attempt_id": turn.attempt_id,
                "project_name": turn.project_name,
                "passed": turn.passed,
                "score": turn.score,
                "primary_error_type": turn.primary_error_type,
                "concept_id": turn.concept_id,
                "feedback_level": turn.feedback_level,
                "feedback_text": turn.feedback_text,
                "user_text": turn.user_text,
                "diff_summary": turn.diff_summary,
                "highlight_node_ids": turn.highlight_node_ids,
                "need_rag": turn.need_rag,
                "need_repair_validation": turn.need_repair_validation,
                "created_at": _iso(now),
            },
            created_at=now,
        )
        return next_id

    def list_agent_turns(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[S.AgentTurnRecord]:
        rows = self._query_student("agent_turn", student_id)
        payloads = [_payload(row) for row in rows]
        if session_id:
            payloads = [p for p in payloads if p.get("session_id") == session_id]
        if task_id:
            payloads = [p for p in payloads if p.get("task_id") == task_id]
        payloads.sort(key=lambda item: int(item.get("id") or 0), reverse=True)
        payloads = payloads[: max(1, limit)]
        payloads.reverse()
        return [
            S.AgentTurnRecord(
                id=int(p["id"]),
                session_id=p.get("session_id") or "",
                task_id=p.get("task_id") or "",
                attempt_id=p.get("attempt_id"),
                project_name=p.get("project_name") or "",
                passed=bool(p.get("passed")),
                score=float(p.get("score") or 0.0),
                primary_error_type=p.get("primary_error_type") or "",
                concept_id=p.get("concept_id"),
                feedback_level=int(p.get("feedback_level") or 1),
                feedback_text=p.get("feedback_text") or "",
                user_text=p.get("user_text") or "",
                diff_summary=p.get("diff_summary") or "",
                highlight_node_ids=list(p.get("highlight_node_ids") or []),
                need_rag=bool(p.get("need_rag")),
                need_repair_validation=bool(p.get("need_repair_validation")),
                created_at=_parse_iso(p["created_at"]),
            )
            for p in payloads
        ]

    def append_affective_event(
        self,
        student_id: str,
        event: S.AffectiveEventWrite,
    ) -> int:
        next_id = self._next_payload_id("affective_event", student_id)
        now = _utc_now()
        self._upsert_record(
            "affective_event",
            pk=self._pk("affective_event", student_id, str(next_id)),
            student_id=student_id,
            record_key=f"{next_id:020d}",
            payload={
                "id": next_id,
                "session_id": event.session_id,
                "task_id": event.task_id,
                "source": event.source,
                "frustration_score": event.frustration_score,
                "confidence_score": event.confidence_score,
                "engagement_score": event.engagement_score,
                "confusion_score": event.confusion_score,
                "evidence": event.evidence,
                "created_at": _iso(now),
            },
            created_at=now,
        )
        return next_id

    def list_affective_events(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[S.AffectiveEventRecord]:
        rows = self._query_student("affective_event", student_id)
        payloads = [_payload(row) for row in rows]
        if session_id:
            payloads = [p for p in payloads if p.get("session_id") == session_id]
        if task_id:
            payloads = [p for p in payloads if p.get("task_id") == task_id]
        payloads.sort(key=lambda item: int(item.get("id") or 0), reverse=True)
        payloads = payloads[: max(1, limit)]
        payloads.reverse()
        return [
            S.AffectiveEventRecord(
                id=int(p["id"]),
                session_id=p.get("session_id") or "",
                task_id=p.get("task_id") or "",
                source=p.get("source") or "agent_inferred",
                frustration_score=float(p.get("frustration_score") or 0.0),
                confidence_score=float(p.get("confidence_score") or 0.5),
                engagement_score=float(p.get("engagement_score") or 0.5),
                confusion_score=float(p.get("confusion_score") or 0.0),
                evidence=p.get("evidence") or "",
                created_at=_parse_iso(p["created_at"]),
            )
            for p in payloads
        ]


__all__ = ["MilvusStore"]
