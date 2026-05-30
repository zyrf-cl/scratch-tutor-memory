"""Orchestrator for the context-management feature.

:class:`ContextService` is the only object the context HTTP router talks
to. It wires a :class:`ContextStore` to a :class:`Compactor` and owns the
windowing policy defaults. It is deliberately independent of the parent
:class:`MemoryService` — the two share no state.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .compaction import Compactor, TruncatingCompactor
from .schemas import (
    AppendResult,
    ContextMessage,
    ContextStats,
    ContextWindow,
)
from .store import ContextStore, SQLiteContextStore

logger = logging.getLogger(__name__)


class ContextService:
    """Routes messages into a context thread and serves bounded windows."""

    def __init__(
        self,
        store: ContextStore,
        compactor: Compactor,
        *,
        default_max_tokens: Optional[int] = None,
        default_max_turns: Optional[int] = None,
    ) -> None:
        self._store = store
        self._compactor = compactor
        self._default_max_tokens = default_max_tokens
        self._default_max_turns = default_max_turns

    def append(
        self,
        student_id: str,
        session_id: str,
        messages: list[ContextMessage],
    ) -> AppendResult:
        records = self._store.append_messages(student_id, session_id, messages)
        stats = self._store.stats(student_id, session_id)
        logger.debug(
            "context_append student=%s session=%s appended=%d total=%d",
            student_id,
            session_id,
            len(records),
            stats.message_count,
        )
        return AppendResult(
            appended=len(records),
            total_messages=stats.message_count,
            total_tokens=stats.total_tokens,
        )

    def get_window(
        self,
        student_id: str,
        session_id: str,
        *,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> ContextWindow:
        # An explicit request value wins; otherwise fall back to the
        # service-level default (which may also be None = unbounded).
        max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        max_turns = max_turns if max_turns is not None else self._default_max_turns

        messages = self._store.list_messages(student_id, session_id)
        pinned = self._store.list_pinned(student_id, session_id)
        result = self._compactor.compact(
            messages,
            max_tokens=max_tokens,
            max_turns=max_turns,
            pinned=pinned,
        )
        return ContextWindow(
            student_id=student_id,
            session_id=session_id,
            messages=result.messages,
            total_messages=len(messages),
            returned_messages=len(result.messages),
            total_tokens=sum(m.token_count for m in result.messages),
            truncated=result.truncated,
            compaction_note=result.note,
        )

    def clear(self, student_id: str, session_id: str) -> int:
        deleted = self._store.clear(student_id, session_id)
        logger.debug(
            "context_clear student=%s session=%s deleted=%d",
            student_id,
            session_id,
            deleted,
        )
        return deleted

    def stats(self, student_id: str, session_id: str) -> ContextStats:
        return self._store.stats(student_id, session_id)


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return None
    return value if value > 0 else None


def build_default_context_service(db_path: Optional[str] = None) -> ContextService:
    """Wire the default context service.

    DB path resolution (first non-empty wins): the ``db_path`` argument,
    then ``MEMORY_MODULE_CONTEXT_DB``, then ``MEMORY_MODULE_DB``, finally
    ``":memory:"``. Sharing ``MEMORY_MODULE_DB`` keeps the context table
    co-located with the rest of the module by default while remaining a
    logically separate store; set ``MEMORY_MODULE_CONTEXT_DB`` to split it
    onto its own file.
    """
    if not db_path:
        db_path = (
            os.environ.get("MEMORY_MODULE_CONTEXT_DB")
            or os.environ.get("MEMORY_MODULE_DB")
            or ":memory:"
        )
    store = SQLiteContextStore(db_path=db_path)
    return ContextService(
        store=store,
        compactor=TruncatingCompactor(),
        default_max_tokens=_env_int("MEMORY_MODULE_CONTEXT_MAX_TOKENS"),
        default_max_turns=_env_int("MEMORY_MODULE_CONTEXT_MAX_TURNS"),
    )


__all__ = ["ContextService", "build_default_context_service"]
