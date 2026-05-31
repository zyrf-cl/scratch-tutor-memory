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

from .compaction import (
    Compactor,
    KimiContextSummarizer,
    LLMSummarizingCompactor,
    TruncatingCompactor,
)
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


class UserContextService:
    """Cross-session, per-user working context.

    A standalone sibling of :class:`ContextService` for context that should
    persist across *all* of a user's sessions (durable goals, profile,
    stable prompt hints). Fully decoupled: its own store/table and its own
    instance, sharing no state with the per-session service or with
    MemoryService. It only reuses the windowing/compaction logic and the
    message schema. Keyed by ``student_id`` alone — there is no session
    dimension in its public surface.
    """

    # Backed by its own table whose session_id column is unused; a fixed
    # empty scope is an internal detail, never exposed to callers.
    _SCOPE = ""

    def __init__(
        self,
        store: ContextStore,
        compactor: Compactor,
        *,
        default_max_tokens: Optional[int] = None,
        default_max_turns: Optional[int] = None,
    ) -> None:
        self._inner = ContextService(
            store,
            compactor,
            default_max_tokens=default_max_tokens,
            default_max_turns=default_max_turns,
        )

    def append(self, student_id: str, messages: list[ContextMessage]) -> AppendResult:
        return self._inner.append(student_id, self._SCOPE, messages)

    def get_window(
        self,
        student_id: str,
        *,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> ContextWindow:
        window = self._inner.get_window(
            student_id, self._SCOPE, max_tokens=max_tokens, max_turns=max_turns
        )
        window.session_id = None  # the user stream has no session
        return window

    def stats(self, student_id: str) -> ContextStats:
        result = self._inner.stats(student_id, self._SCOPE)
        result.session_id = None
        return result

    def clear(self, student_id: str) -> int:
        return self._inner.clear(student_id, self._SCOPE)


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


def _resolve_db_path(db_path: Optional[str]) -> str:
    """DB path resolution (first non-empty wins): the ``db_path`` argument,
    then ``MEMORY_MODULE_CONTEXT_DB``, then ``MEMORY_MODULE_DB``, finally
    ``":memory:"``. Sharing ``MEMORY_MODULE_DB`` keeps the context tables
    co-located with the rest of the module by default while remaining
    logically separate stores; set ``MEMORY_MODULE_CONTEXT_DB`` to split
    them onto their own file."""
    return (
        db_path
        or os.environ.get("MEMORY_MODULE_CONTEXT_DB")
        or os.environ.get("MEMORY_MODULE_DB")
        or ":memory:"
    )


def build_default_context_service(db_path: Optional[str] = None) -> ContextService:
    """Wire the default per-session context service (table ``context_message``)."""
    store = SQLiteContextStore(db_path=_resolve_db_path(db_path))
    return ContextService(
        store=store,
        compactor=_build_compactor(),
        default_max_tokens=_env_int("MEMORY_MODULE_CONTEXT_MAX_TOKENS"),
        default_max_turns=_env_int("MEMORY_MODULE_CONTEXT_MAX_TURNS"),
    )


def build_default_user_context_service(
    db_path: Optional[str] = None,
) -> UserContextService:
    """Wire the default cross-session per-user context service.

    Same DB resolution as :func:`build_default_context_service`, but stored
    in its own ``user_context_message`` table so it shares no state with the
    per-session context stream (or with MemoryService)."""
    store = SQLiteContextStore(
        db_path=_resolve_db_path(db_path), table="user_context_message"
    )
    return UserContextService(
        store=store,
        compactor=_build_compactor(),
        default_max_tokens=_env_int("MEMORY_MODULE_CONTEXT_MAX_TOKENS"),
        default_max_turns=_env_int("MEMORY_MODULE_CONTEXT_MAX_TURNS"),
    )


def _build_compactor() -> Compactor:
    """Pick the windowing strategy.

    With ``MEMORY_MODULE_USE_KIMI=1`` *and* a Kimi credential
    (``KIMI_API_KEY`` or ``MOONSHOT_API_KEY``) set, summarize the overflow
    via :class:`LLMSummarizingCompactor` (backed by Kimi/Moonshot);
    otherwise drop it with the dependency-free
    :class:`TruncatingCompactor`. Construction failure degrades to
    truncation so the feature never hard-fails on wiring.
    """
    use_kimi = os.environ.get("MEMORY_MODULE_USE_KIMI", "0").lower() in {
        "1",
        "true",
        "yes",
    } and bool(os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY"))
    if not use_kimi:
        return TruncatingCompactor()
    try:
        compactor = LLMSummarizingCompactor(KimiContextSummarizer())
        logger.info("context: using Kimi LLM-summarizing compactor")
        return compactor
    except Exception as exc:  # noqa: BLE001
        logger.warning("context: Kimi compactor unavailable, truncating: %s", exc)
        return TruncatingCompactor()


__all__ = [
    "ContextService",
    "build_default_context_service",
    "UserContextService",
    "build_default_user_context_service",
]
