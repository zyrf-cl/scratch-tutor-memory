"""Extractors that turn raw inputs into the C (dialog summary) and
F (misconception cluster) memory types.

Both are :class:`Protocol`s so the LLM-backed implementations can drop
in later without touching the service layer. The defaults bundled
here are deterministic, dependency-free stubs:

* :class:`StubDialogSummarizer` — concatenates the student's utterances
  and trims to a length cap. Good enough to demonstrate that "talking
  about X" is later recoverable through the dialog channel.

* :class:`StubMisconceptionDetector` — counts errors per concept and
  promotes any concept with ≥ ``threshold`` occurrences to a
  misconception cluster.

LLM-backed alternatives live below the stubs:

* :class:`MiMoDialogSummarizer` — calls MiMo to write a
  pedagogy-focused Chinese summary of a tutoring session.

* :class:`MiMoMisconceptionDetector` — calls MiMo to cluster a list
  of observed errors into named misconceptions with a human-readable
  description; falls back to per-concept counting on parse failure.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Protocol

from pydantic import BaseModel

from . import schemas as S
from .mimo_client import MiMoClient

logger = logging.getLogger(__name__)


class MisconceptionCluster(BaseModel):
    pattern_id: str
    description: str


class DialogSummarizer(Protocol):
    def summarize(self, turns: list[S.DialogTurn]) -> str: ...


class MisconceptionDetector(Protocol):
    def detect(self, errors: list[dict[str, Any]]) -> list[MisconceptionCluster]: ...


class StubDialogSummarizer:
    """Naive 'summary' = pipe-joined student utterances, capped at 500 chars."""

    def __init__(self, max_chars: int = 500) -> None:
        self._max = max_chars

    def summarize(self, turns: list[S.DialogTurn]) -> str:
        students = [t.content for t in turns if t.role == "student"]
        if not students:
            return "(empty session)"
        joined = " | ".join(students)
        return joined[: self._max]


class StubMisconceptionDetector:
    """Promote any concept with >= threshold error events into a misconception."""

    def __init__(self, threshold: int = 2) -> None:
        self._threshold = threshold

    def detect(self, errors: list[dict[str, Any]]) -> list[MisconceptionCluster]:
        counts: Counter[str] = Counter(
            e.get("concept_id") for e in errors if e.get("concept_id")
        )
        clusters: list[MisconceptionCluster] = []
        for concept, n in counts.items():
            if n >= self._threshold:
                clusters.append(
                    MisconceptionCluster(
                        pattern_id=f"concept:{concept}",
                        description=(
                            f"Repeatedly struggles with '{concept}' "
                            f"— observed {n} times"
                        ),
                    )
                )
        return clusters


__all__ = [
    "MisconceptionCluster",
    "DialogSummarizer",
    "MisconceptionDetector",
    "StubDialogSummarizer",
    "StubMisconceptionDetector",
    "MiMoDialogSummarizer",
    "MiMoMisconceptionDetector",
]


# ---------------- MiMo-backed implementations ----------------


_SUMMARIZER_SYSTEM = (
    "你是 Scratch 编程教学的助教记忆官。"
    "把一段师生对话压成一句话长摘要，"
    "突出学生关心的目标、卡住的地方、用到的积木。"
    "不要解释思考过程，只输出最终摘要本身，不超过 80 个汉字。"
)


class MiMoDialogSummarizer:
    """Calls MiMo to produce a single-line Chinese summary of a session."""

    def __init__(
        self,
        client: MiMoClient | None = None,
        max_chars: int = 200,
        temperature: float = 0.2,
    ) -> None:
        self._client = client or MiMoClient()
        self._max = max_chars
        self._temperature = temperature

    def summarize(self, turns: list[S.DialogTurn]) -> str:
        if not turns:
            return "(empty session)"
        transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
        try:
            text = self._client.chat(
                messages=[
                    {"role": "system", "content": _SUMMARIZER_SYSTEM},
                    {"role": "user", "content": f"对话：\n{transcript}\n\n请给出一句话摘要。"},
                ],
                temperature=self._temperature,
                max_tokens=3072,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MiMo summarize failed, falling back: %s", exc)
            students = [t.content for t in turns if t.role == "student"]
            return (" | ".join(students) or "(empty session)")[: self._max]
        text = text.replace("\n", " ").strip()
        return text[: self._max] if text else "(empty session)"


_DETECTOR_SYSTEM = (
    "你是 Scratch 教学的迷思诊断官。"
    "给定一批 error_observed 事件（每条带 concept_id 与 description），"
    "把它们按潜在迷思聚类。"
    "只输出 JSON 数组，每个元素形如 "
    '{"pattern_id":"<英文小写蛇形>","description":"<一句话中文描述>","concept_ids":["loop"]}。'
    "只输出 JSON 本身，不要任何解释、不要 markdown 代码块包裹。"
    "如果没有可成立的迷思，输出 []。"
)


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _extract_json_array(text: str) -> Any:
    """Best-effort grab of the first JSON array in ``text``."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        raise ValueError("no JSON array in model response")
    return json.loads(m.group(0))


class MiMoMisconceptionDetector:
    """LLM-backed misconception clustering with a deterministic fallback.

    Sends the full list of observed errors to MiMo and asks for a JSON
    array of clusters. If parsing fails or the model returns nothing,
    falls back to :class:`StubMisconceptionDetector` so the demo always
    has *some* signal in channel F.
    """

    def __init__(
        self,
        client: MiMoClient | None = None,
        threshold: int = 2,
        temperature: float = 0.1,
    ) -> None:
        self._client = client or MiMoClient()
        self._fallback = StubMisconceptionDetector(threshold=threshold)
        self._temperature = temperature

    def detect(self, errors: list[dict[str, Any]]) -> list[MisconceptionCluster]:
        if not errors:
            return []
        payload = [
            {
                "concept_id": e.get("concept_id"),
                "description": e.get("description") or "",
            }
            for e in errors
        ]
        try:
            text = self._client.chat(
                messages=[
                    {"role": "system", "content": _DETECTOR_SYSTEM},
                    {
                        "role": "user",
                        "content": "错误事件：\n" + json.dumps(payload, ensure_ascii=False),
                    },
                ],
                temperature=self._temperature,
                max_tokens=3072,
            )
            raw = _extract_json_array(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MiMo detect failed, using stub fallback: %s", exc)
            return self._fallback.detect(errors)

        clusters: list[MisconceptionCluster] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("pattern_id") or "").strip()
            desc = str(item.get("description") or "").strip()
            if not pid or not desc or pid in seen:
                continue
            seen.add(pid)
            clusters.append(MisconceptionCluster(pattern_id=pid, description=desc))

        if not clusters:
            # Model returned [] or pure garbage — fall back so F channel
            # is non-empty when there are repeated errors.
            return self._fallback.detect(errors)
        return clusters
