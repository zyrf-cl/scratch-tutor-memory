"""High-level orchestrator for the memory module.

:class:`MemoryService` is the only object the API layer talks to.
It owns the wiring between :class:`MemoryStore`, :class:`Embedder`
and the two extractors, and centralises the policy decisions
(when to re-cluster misconceptions, what counts as a "session end",
how to merge mastery evidence).

Use :func:`build_default_service` for the demo defaults, or wire
your own pieces if you need a non-stub LLM extractor or a Postgres
store.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from . import schemas as S
from .concepts import is_valid_concept
from .embedding import Embedder, build_embedder
from .extractor import (
    AffectiveSignalDetector,
    DialogSummarizer,
    MiMoDialogSummarizer,
    MiMoAffectiveSignalDetector,
    MiMoMisconceptionDetector,
    MisconceptionDetector,
    StubAffectiveSignalDetector,
    StubDialogSummarizer,
    StubMisconceptionDetector,
)
from .store import MemoryStore, SQLiteStore

logger = logging.getLogger(__name__)


def _build_store(db_path: str, *, vector_dim: int) -> MemoryStore:
    store_name = os.environ.get("MEMORY_MODULE_STORE", "sqlite").strip().lower()
    if store_name in {"", "sqlite"}:
        return SQLiteStore(db_path=db_path)
    if store_name == "milvus":
        from .milvus_store import MilvusStore

        return MilvusStore(vector_dim=vector_dim)
    raise ValueError(
        f"Unknown MEMORY_MODULE_STORE: {store_name!r} "
        "(expected 'sqlite' or 'milvus')"
    )


class MemoryService:
    """Routes events / dialog into the appropriate memory channels and
    aggregates the read-side state."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        dialog_summarizer: DialogSummarizer,
        misconception_detector: MisconceptionDetector,
        affective_signal_detector: AffectiveSignalDetector,
        misconception_refresh_every: int = 1,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._summarizer = dialog_summarizer
        self._detector = misconception_detector
        self._affective = affective_signal_detector
        # Re-run the misconception detector after every N error events.
        # Default 1 = re-run on every error, which is fine for demo scale.
        self._refresh_every = max(1, misconception_refresh_every)

    # ----- write paths -----

    def handle_event(self, student_id: str, event: S.Event) -> None:
        if isinstance(event, S.MasteryUpdateEvent):
            if not is_valid_concept(event.concept_id):
                raise ValueError(f"unknown concept_id: {event.concept_id}")
            self._store.upsert_mastery(
                student_id, event.concept_id, event.score, event.evidence_note
            )
            logger.debug(
                "mastery_update student=%s concept=%s score=%.2f",
                student_id,
                event.concept_id,
                event.score,
            )
            return

        if isinstance(event, S.SessionEndEvent):
            self._store.update_session(
                student_id,
                last_session_at=datetime.now(timezone.utc),
                current_project_id=event.current_project_id,
                current_topic=event.current_topic,
                pending_task=event.pending_task,
            )
            logger.debug("session_end student=%s topic=%s", student_id, event.current_topic)
            return

        if isinstance(event, S.ProjectSavedEvent):
            self._store.save_project(student_id, event)
            logger.debug(
                "project_saved student=%s project=%s", student_id, event.project_id
            )
            return

        if isinstance(event, S.PreferenceSetEvent):
            self._store.set_preference(student_id, event)
            logger.debug(
                "preference_set student=%s key=%s", student_id, event.key
            )
            return

        if isinstance(event, S.ErrorObservedEvent):
            if event.concept_id is not None and not is_valid_concept(event.concept_id):
                raise ValueError(f"unknown concept_id: {event.concept_id}")
            self._store.append_error(student_id, event)
            errors = self._store.list_errors(student_id)
            if len(errors) % self._refresh_every == 0:
                self._refresh_misconceptions(student_id, errors)
            logger.debug(
                "error_observed student=%s concept=%s", student_id, event.concept_id
            )
            return

        # The discriminated union should make this unreachable, but keep
        # an explicit guard so silent drops never happen.
        raise ValueError(f"Unhandled event type: {type(event).__name__}")

    def handle_dialog(self, student_id: str, turns: list[S.DialogTurn]) -> int:
        summary = self._summarizer.summarize(turns)
        emb = self._embedder.embed(summary).tolist()
        turn_id = self._store.append_dialog_summary(student_id, summary, emb)
        logger.debug(
            "dialog_summary student=%s turn=%d len=%d",
            student_id,
            turn_id,
            len(summary),
        )
        return turn_id

    def _refresh_misconceptions(
        self, student_id: str, errors: list[dict[str, Any]]
    ) -> None:
        clusters = self._detector.detect(errors)
        for cluster in clusters:
            emb = self._embedder.embed(cluster.description).tolist()
            self._store.upsert_misconception(
                student_id,
                cluster.pattern_id,
                cluster.description,
                emb,
                occurrences=cluster.occurrences,
                last_seen=cluster.last_seen,
            )

    # ----- read paths -----

    def get_state(self, student_id: str) -> S.StudentState:
        return S.StudentState(
            student_id=student_id,
            mastery=self._store.get_mastery(student_id),
            session=self._store.get_session(student_id),
            projects=self._store.list_projects(student_id),
            preferences=self._store.list_preferences(student_id),
            misconceptions=self._store.list_misconceptions(student_id),
        )

    def recall(
        self,
        student_id: str,
        query: str,
        kind: str = "dialog",
        top_k: int = 5,
    ) -> list[S.RecallHit]:
        emb = self._embedder.embed(query).tolist()
        if kind == "dialog":
            pairs = self._store.search_dialog(student_id, emb, top_k)
            return [
                S.RecallHit(
                    kind="dialog", score=score, payload=rec.model_dump(mode="json")
                )
                for rec, score in pairs
            ]
        if kind == "misconception":
            pairs = self._store.search_misconceptions(student_id, emb, top_k)
            return [
                S.RecallHit(
                    kind="misconception",
                    score=score,
                    payload=rec.model_dump(mode="json"),
                )
                for rec, score in pairs
            ]
        raise ValueError(
            f"Unknown recall kind: {kind!r} (expected 'dialog' or 'misconception')"
        )

    # ----- PracticeAgent memory integration -----

    def handle_agent_turn(self, student_id: str, turn: S.AgentTurnWrite) -> int:
        if turn.concept_id is not None and not is_valid_concept(turn.concept_id):
            raise ValueError(f"unknown concept_id: {turn.concept_id}")

        history = self._store.list_agent_turns(
            student_id,
            task_id=turn.task_id or None,
            limit=10,
        )
        turn_id = self._store.append_agent_turn(student_id, turn)
        affective_id = self._store.append_affective_event(
            student_id,
            self._affective.detect(turn=turn, history=history),
        )

        # Mirror failed diagnostic turns into the F channel so the
        # existing misconception machinery can still surface long-term
        # conceptual patterns.
        if (
            not turn.passed
            and turn.concept_id
            and (turn.primary_error_type or turn.diff_summary)
        ):
            self._store.append_error(
                student_id,
                S.ErrorObservedEvent(
                    concept_id=turn.concept_id,
                    description=turn.diff_summary or turn.primary_error_type,
                    context=(
                        f"task_id={turn.task_id}; session_id={turn.session_id}; "
                        f"error_type={turn.primary_error_type}"
                    ),
                ),
            )
            errors = self._store.list_errors(student_id)
            if len(errors) % self._refresh_every == 0:
                self._refresh_misconceptions(student_id, errors)

        logger.debug(
            "agent_turn student=%s task=%s error=%s level=%d affective_event=%d",
            student_id,
            turn.task_id,
            turn.primary_error_type,
            turn.feedback_level,
            affective_id,
        )
        return turn_id

    def get_agent_memory_snapshot(
        self,
        student_id: str,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> S.AgentMemorySnapshot:
        task_turns = (
            self._store.list_agent_turns(student_id, task_id=task_id, limit=10)
            if task_id
            else []
        )
        recent_turns = task_turns or self._store.list_agent_turns(
            student_id,
            session_id=session_id,
            limit=10,
        )
        if not recent_turns:
            recent_turns = self._store.list_agent_turns(student_id, limit=10)

        feedbacks = [
            t.feedback_text[:200]
            for t in recent_turns
            if t.feedback_text
        ][-5:]
        error_types = [
            t.primary_error_type
            for t in recent_turns
            if t.primary_error_type
        ][-10:]
        recent_level = recent_turns[-1].feedback_level if recent_turns else 1

        if error_types:
            current_error = error_types[-1]
            repeated_count = max(0, error_types.count(current_error) - 1)
        else:
            repeated_count = 0

        mastered = [
            rec.concept_id
            for rec in self._store.get_mastery(student_id)
            if rec.score >= 0.75
        ]

        all_turns = self._store.list_agent_turns(student_id, limit=100)
        weak_counts: Counter[str] = Counter(
            t.concept_id
            for t in all_turns
            if t.concept_id and not t.passed
        )
        weak = [concept for concept, count in weak_counts.items() if count >= 2]
        for misconception in self._store.list_misconceptions(student_id):
            if misconception.pattern_id.startswith("concept:"):
                concept = misconception.pattern_id.split(":", 1)[1]
                if concept not in weak:
                    weak.append(concept)

        return S.AgentMemorySnapshot(
            recent_feedbacks=feedbacks,
            recent_error_types=error_types,
            recent_feedback_level=recent_level,
            repeated_error_count=repeated_count,
            mastered_concepts=mastered,
            weak_concepts=weak,
            affective_state=self._aggregate_affective_state(
                student_id,
                session_id=session_id,
                task_id=task_id,
            ),
        )

    def handle_affective_event(
        self,
        student_id: str,
        event: S.AffectiveEventWrite,
    ) -> int:
        event_id = self._store.append_affective_event(student_id, event)
        logger.debug(
            "affective_event student=%s task=%s frustration=%.2f confidence=%.2f",
            student_id,
            event.task_id,
            event.frustration_score,
            event.confidence_score,
        )
        return event_id

    def _aggregate_affective_state(
        self,
        student_id: str,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, float]:
        events = (
            self._store.list_affective_events(student_id, task_id=task_id, limit=10)
            if task_id
            else []
        )
        if not events:
            events = self._store.list_affective_events(
                student_id,
                session_id=session_id,
                limit=10,
            )
        if not events:
            return {
                "frustration": 0.0,
                "confidence": 0.5,
                "engagement": 0.5,
                "confusion": 0.0,
            }

        total_weight = 0.0
        sums = {
            "frustration": 0.0,
            "confidence": 0.0,
            "engagement": 0.0,
            "confusion": 0.0,
        }
        for idx, event in enumerate(events, start=1):
            weight = float(idx)
            total_weight += weight
            sums["frustration"] += event.frustration_score * weight
            sums["confidence"] += event.confidence_score * weight
            sums["engagement"] += event.engagement_score * weight
            sums["confusion"] += event.confusion_score * weight

        return {
            key: round(value / total_weight, 3)
            for key, value in sums.items()
        }


def build_default_service(
    db_path: str = ":memory:",
    use_mimo: bool | None = None,
    embedder_name: str | None = None,
) -> MemoryService:
    """Wire up the demo defaults.

    If ``use_mimo`` is True (or ``MEMORY_MODULE_USE_MIMO=1`` is set in
    the env *and* ``MIMO_API_KEY`` is present), swap the stub
    summarizer and misconception detector for MiMo-backed ones. The
    store and embedder stay deterministic so semantic recall remains
    reproducible across runs."""
    embedder = build_embedder(embedder_name)
    store: MemoryStore = _build_store(db_path=db_path, vector_dim=embedder.dim)

    if use_mimo is None:
        use_mimo = (
            os.environ.get("MEMORY_MODULE_USE_MIMO", "0").lower() in {"1", "true", "yes"}
            and bool(os.environ.get("MIMO_API_KEY"))
        )

    summarizer: DialogSummarizer
    detector: MisconceptionDetector
    affective_detector: AffectiveSignalDetector
    if use_mimo:
        logger.info("memory_module: using MiMo extractor")
        summarizer = MiMoDialogSummarizer()
        detector = MiMoMisconceptionDetector(threshold=2)
        affective_detector = MiMoAffectiveSignalDetector()
    else:
        summarizer = StubDialogSummarizer()
        detector = StubMisconceptionDetector(threshold=2)
        affective_detector = StubAffectiveSignalDetector()

    return MemoryService(
        store=store,
        embedder=embedder,
        dialog_summarizer=summarizer,
        misconception_detector=detector,
        affective_signal_detector=affective_detector,
    )


__all__ = ["MemoryService", "build_default_service"]
