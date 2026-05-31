"""Compaction strategy for the context window.

A :class:`Compactor` decides which messages survive when the agent
asks for a bounded window. Two implementations ship:

* :class:`TruncatingCompactor` (default) — pure recency windowing (drop
  the oldest non-pinned messages that do not fit the budget). Never
  calls an LLM, so the default path stays dependency-free.

* :class:`LLMSummarizingCompactor` — instead of dropping the overflow,
  it summarizes those older messages into one synthetic ``system``
  message via a :class:`ContextSummarizer`, prepends it, and records a
  ``CompactionResult.note``. Same signature as the Protocol, so neither
  the service nor the HTTP API changes; it is selected at the wiring
  layer (``MEMORY_MODULE_USE_KIMI=1``, backed by Kimi/Moonshot) and
  falls back to plain truncation if summarization fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from .schemas import ContextMessageRecord, estimate_tokens

logger = logging.getLogger(__name__)


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


# Sentinel id for the synthetic summary record (never collides with a
# real AUTOINCREMENT rowid, which is always >= 1).
_SUMMARY_ID = -1


class ContextSummarizer(Protocol):
    """Summarizes a run of overflow messages into one short note."""

    def summarize(self, messages: list[ContextMessageRecord]) -> str: ...


class LLMSummarizingCompactor:
    """Recency windowing that summarizes the overflow instead of dropping it.

    It first runs :class:`TruncatingCompactor` to pick the recent window
    that fits the budget (reserving ``summary_token_budget`` tokens / one
    turn for the summary). The non-pinned messages that did not make the
    cut are handed to the :class:`ContextSummarizer`; the resulting text
    becomes a single synthetic ``system`` message positioned where the
    overflow used to be, and ``CompactionResult.note`` records what
    happened. If nothing overflowed, the LLM is never called. If
    summarization raises, it degrades to the plain truncation result.
    """

    def __init__(
        self,
        summarizer: ContextSummarizer,
        *,
        summary_token_budget: int = 256,
    ) -> None:
        self._summarizer = summarizer
        self._summary_token_budget = summary_token_budget
        self._truncator = TruncatingCompactor()

    def compact(
        self,
        messages: list[ContextMessageRecord],
        *,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        pinned: Optional[list[ContextMessageRecord]] = None,
    ) -> CompactionResult:
        # Reserve room for the summary message so the final window
        # (window + summary) still respects the caller's budget.
        eff_max_tokens = (
            None
            if max_tokens is None
            else max(1, max_tokens - self._summary_token_budget)
        )
        eff_max_turns = None if max_turns is None else max(1, max_turns - 1)

        base = self._truncator.compact(
            messages,
            max_tokens=eff_max_tokens,
            max_turns=eff_max_turns,
            pinned=pinned,
        )
        if not base.truncated:
            return base  # everything fit — no overflow, no LLM call

        kept_ids = {m.id for m in base.messages}
        dropped = [m for m in messages if m.id not in kept_ids]
        try:
            summary = self._summarizer.summarize(dropped).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("context summarize failed, plain truncation: %s", exc)
            return base
        if not summary:
            return base

        note = f"compacted {len(dropped)} older message(s) into a summary"
        record = ContextMessageRecord(
            id=_SUMMARY_ID,
            seq=dropped[0].seq,  # stand in for the overflow's position
            role="system",
            content=summary,
            token_count=estimate_tokens(summary),
            created_at=dropped[0].created_at,
            metadata={"synthetic": True, "compacted_messages": len(dropped)},
        )
        merged = sorted(base.messages + [record], key=lambda m: m.seq)
        return CompactionResult(messages=merged, truncated=True, note=note)


_CONTEXT_SUMMARY_SYSTEM = (
    "你是对话上下文压缩器。把下面这段较早的对话压成简洁要点，"
    "保留关键事实、已做的决定、未完成的任务与用户偏好，省略寒暄与冗余。"
    "只输出摘要本身，不要解释思考过程，不超过 200 字。"
)


class KimiContextSummarizer:
    """:class:`ContextSummarizer` backed by Kimi (Moonshot, OpenAI-compatible)."""

    def __init__(self, client=None, max_chars: int = 600, temperature: float = 0.2) -> None:
        if client is None:
            # Deferred import: avoids pulling the client (and httpx) when
            # the default (truncating) compactor is used.
            from ..kimi_client import KimiClient

            client = KimiClient()
        self._client = client
        self._max = max_chars
        self._temperature = temperature

    def summarize(self, messages: list[ContextMessageRecord]) -> str:
        if not messages:
            return ""
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        text = self._client.chat(
            messages=[
                {"role": "system", "content": _CONTEXT_SUMMARY_SYSTEM},
                {"role": "user", "content": f"较早的对话：\n{transcript}\n\n请压缩为要点摘要。"},
            ],
            temperature=self._temperature,
            max_tokens=1024,
        )
        return text.replace("\n", " ").strip()[: self._max]


__all__ = [
    "CompactionResult",
    "Compactor",
    "TruncatingCompactor",
    "ContextSummarizer",
    "LLMSummarizingCompactor",
    "KimiContextSummarizer",
]
