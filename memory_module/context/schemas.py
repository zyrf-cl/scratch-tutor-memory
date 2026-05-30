"""Pydantic models for the decoupled context-management feature.

This is *short-term working context* — the running conversation the
agent pushes in and reads back a bounded window of to feed its LLM.
It is intentionally separate from the long-term ``dialog`` channel
(channel C), which summarizes + embeds turns for semantic recall.

Conventions match the parent module: timezone-aware UTC datetimes,
``__all__`` for the public surface, ``from __future__ import annotations``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# OpenAI-style chat roles — this is the agent's LLM context, not the
# teaching dialog (which uses agent/student). Kept distinct on purpose.
Role = Literal["system", "user", "assistant", "tool"]


def estimate_tokens(text: str) -> int:
    """Rough, dependency-free token estimate.

    Used only as a fallback when the caller does not supply an exact
    ``token_count``. The ``len // 4`` heuristic is deliberately crude
    (it over-counts CJK, under-counts code); agents that care about
    precise budgeting should pass ``token_count`` per message.
    """
    return max(1, round(len(text) / 4))


# ===== write path =====


class ContextMessage(BaseModel):
    """One message the agent pushes into a context thread."""

    role: Role
    content: str
    name: Optional[str] = None
    # Caller-supplied exact token count; if omitted we estimate on write.
    token_count: Optional[int] = Field(default=None, ge=0)
    # Pinned messages (e.g. the system prompt) are never dropped by the
    # windowing compactor, regardless of budget.
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppendContextRequest(BaseModel):
    messages: list[ContextMessage]


# ===== read path =====


class ContextMessageRecord(ContextMessage):
    """A persisted message. ``token_count`` is always resolved here."""

    id: int
    seq: int
    token_count: int = Field(ge=0)
    created_at: datetime


class ContextWindow(BaseModel):
    """The bounded working window returned to the agent."""

    student_id: str
    session_id: str
    messages: list[ContextMessageRecord] = Field(default_factory=list)
    total_messages: int = 0
    returned_messages: int = 0
    total_tokens: int = 0
    truncated: bool = False
    # Reserved for a future LLM-summarizing compactor: a synthetic note
    # describing what was compacted away. ``None`` for plain truncation.
    compaction_note: Optional[str] = None


class AppendResult(BaseModel):
    appended: int
    total_messages: int
    total_tokens: int


class ContextStats(BaseModel):
    student_id: str
    session_id: str
    message_count: int = 0
    total_tokens: int = 0
    last_updated: Optional[datetime] = None


__all__ = [
    "Role",
    "estimate_tokens",
    "ContextMessage",
    "AppendContextRequest",
    "ContextMessageRecord",
    "ContextWindow",
    "AppendResult",
    "ContextStats",
]
