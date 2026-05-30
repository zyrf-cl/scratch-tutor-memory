"""Compaction strategy for the context window.

A :class:`Compactor` decides which messages survive when the agent
asks for a bounded window. The default :class:`TruncatingCompactor`
does pure recency windowing (drop the oldest non-pinned messages that
do not fit the budget) and never calls an LLM — keeping this feature
dependency-free and aligned with the v1 "no LLM context management"
boundary.

The Protocol is the seam for a future ``LLMSummarizingCompactor``:
such an implementation would summarize the dropped messages into a
single synthetic ``system``/``assistant`` message, prepend it, and set
``CompactionResult.note`` — all behind the same signature, so neither
the service nor the HTTP API would change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .schemas import ContextMessageRecord


@dataclass
class CompactionResult:
    messages: list[ContextMessageRecord]
    truncated: bool = False
    note: Optional[str] = None


class Compactor(Protocol):
    def compact(
        self,
        messages: list[ContextMessageRecord],
        *,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        pinned: Optional[list[ContextMessageRecord]] = None,
    ) -> CompactionResult: ...


class TruncatingCompactor:
    """Recency windowing under a token / turn budget.

    Policy:

    * ``pinned`` messages are always kept (e.g. the system prompt), even
      if they alone exceed the budget — they are inviolable, so the
      returned window may exceed ``max_tokens``/``max_turns`` when pinned
      content is large.
    * the remaining budget is filled with the **most recent** non-pinned
      messages, walking newest → oldest until either budget is hit.
    * the result is returned in chronological (``seq``) order.
    * with no budget at all, every message is returned unchanged.

    Note: windowing keeps the most recent *contiguous* run of non-pinned
    messages. Walking newest → oldest, the first message that does not fit
    the budget stops the scan — every older message is dropped too, even
    one that would have fit on its own. This avoids handing the LLM a
    conversation with a hole in the middle. Corollary: if the newest
    non-pinned message alone exceeds the budget, no non-pinned message is
    returned. ``truncated`` means "messages were dropped", not "the
    returned window is within budget" (pinned content can still push it
    over).
    """

    def compact(
        self,
        messages: list[ContextMessageRecord],
        *,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        pinned: Optional[list[ContextMessageRecord]] = None,
    ) -> CompactionResult:
        if max_tokens is None and max_turns is None:
            return CompactionResult(messages=list(messages), truncated=False)

        pinned = pinned or []
        pinned_ids = {m.id for m in pinned}
        non_pinned = [m for m in messages if m.id not in pinned_ids]

        token_budget = (
            None
            if max_tokens is None
            else max_tokens - sum(m.token_count for m in pinned)
        )
        turn_budget = None if max_turns is None else max_turns - len(pinned)

        kept_reversed: list[ContextMessageRecord] = []
        used_tokens = 0
        for msg in reversed(non_pinned):
            if turn_budget is not None and len(kept_reversed) >= turn_budget:
                break
            if (
                token_budget is not None
                and used_tokens + msg.token_count > token_budget
            ):
                break
            kept_reversed.append(msg)
            used_tokens += msg.token_count

        kept = list(reversed(kept_reversed))
        truncated = len(kept) < len(non_pinned)
        merged = sorted(pinned + kept, key=lambda m: m.seq)
        return CompactionResult(messages=merged, truncated=truncated)


__all__ = ["CompactionResult", "Compactor", "TruncatingCompactor"]
