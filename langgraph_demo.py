"""LangGraph demo wiring the MiMo LLM into the memory module.

The graph models one student turn at a time:

    START
      ↓
    recall      ← load StudentState + semantic recall from memory_module
      ↓
    respond     ← MiMo writes a tutor reply grounded in memory
      ↓
    extract     ← MiMo turns the student's last message into structured
                  events (mastery_update, error_observed, …)
      ↓
    save        ← write events + dialog summary back into memory_module
      ↓
     END

A scenario runner then drives two sessions for student "alice" to
prove the six memory channels (A–F) survive across sessions:

* Session 1 — fresh start, no memory.  alice declares a preference,
  plays with loops + events, hits two event-related errors, saves a
  project, ends the session.

* Session 2 — agent first calls ``recall``; we *assert* the recalled
  state already contains B (pending task), D (project), E
  (preference), A (mastery), and F (misconception cluster), and that
  semantic recall finds the dialog summary from session 1.

We also confirm bob is isolated from alice across the same MiMo-
backed graph.

Run::

    uv run --env-file .env python langgraph_demo.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Annotated, Any, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env so MIMO_API_KEY / BASE_URL / MODEL are visible to the
# memory_module + MiMo client even when started without --env-file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import json
import re

from langgraph.graph import END, START, StateGraph

from memory_module import schemas as S
from memory_module.mimo_client import MiMoClient
from memory_module.service import MemoryService, build_default_service

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("langgraph_demo")


# ---------------- shared singletons ----------------

# One shared MiMo client + memory service for the whole graph run.
# (In a real product the service would live behind an HTTP boundary;
# here the graph is the only caller, so we wire it in-process.)
os.environ.setdefault("MEMORY_MODULE_USE_MIMO", "1")
SERVICE: MemoryService = build_default_service(db_path=":memory:", use_mimo=True)
MIMO = MiMoClient()


# ---------------- graph state ----------------


class TurnState(TypedDict, total=False):
    student_id: str
    student_utterance: str          # what the student just said
    extra_events: list[dict]        # pre-baked events the harness wants saved
    recalled_state: dict            # snapshot of StudentState
    recalled_dialog: list[dict]     # top-k dialog hits
    agent_reply: str
    extracted_events: list[dict]
    saved_event_count: int


# ---------------- nodes ----------------


def recall_node(state: TurnState) -> TurnState:
    sid = state["student_id"]
    log.info("[%s] recall...", sid)
    st = SERVICE.get_state(sid).model_dump(mode="json")
    hits: list[dict[str, Any]] = []
    if state.get("student_utterance"):
        hits = [
            h.model_dump(mode="json")
            for h in SERVICE.recall(
                sid, query=state["student_utterance"], kind="dialog", top_k=3
            )
        ]
    return {"recalled_state": st, "recalled_dialog": hits}


_RESPOND_SYSTEM = (
    "你是 Scratch 编程教学 agent。"
    "根据学生画像与历史对话续点，用 1–2 句话回应学生。"
    "如果画像里有 pending_task 或 current_topic，要自然地承接，不要装作不记得。"
    "只输出给学生看的回复本身。"
)


def respond_node(state: TurnState) -> TurnState:
    if not state.get("student_utterance"):
        return {"agent_reply": ""}
    log.info("[%s] respond... (MiMo reasoning, ~15-30s)", state["student_id"])

    profile = state.get("recalled_state") or {}
    session = profile.get("session") or {}
    mastery_lines = ", ".join(
        f"{m['concept_id']}={m['score']:.2f}"
        for m in (profile.get("mastery") or [])
    )
    prefs = ", ".join(
        f"{p['key']}={p['value']}" for p in (profile.get("preferences") or [])
    )
    misc = "; ".join(
        f"{m['pattern_id']}({m['occurrences']}次)"
        for m in (profile.get("misconceptions") or [])
    )
    dialog_hits = "\n".join(
        f"- score={h['score']:.2f}: {h['payload'].get('summary','')}"
        for h in (state.get("recalled_dialog") or [])
    )

    profile_text = (
        f"上次会话: project={session.get('current_project_id')}, "
        f"topic={session.get('current_topic')}, pending={session.get('pending_task')}\n"
        f"掌握度: {mastery_lines or '(空)'}\n"
        f"偏好: {prefs or '(空)'}\n"
        f"迷思: {misc or '(空)'}\n"
        f"相关历史对话:\n{dialog_hits or '(无)'}"
    )

    reply = MIMO.chat(
        messages=[
            {"role": "system", "content": _RESPOND_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"=== 学生画像 ===\n{profile_text}\n\n"
                    f"=== 学生当前发言 ===\n{state['student_utterance']}\n\n"
                    "请给出回复。"
                ),
            },
        ],
        temperature=0.4,
        max_tokens=3072,
    )
    return {"agent_reply": reply}


_EXTRACT_SYSTEM = (
    "你是 Scratch 教学的事件抽取器。"
    "从学生这一轮发言中抽出结构化事件。"
    "事件类型只能是: mastery_update / error_observed / preference_set。"
    "concept_id 只能从 [loop, conditional, event, variable, broadcast, clone, sprite, sound, motion, sensing, operator, list] 选。"
    "如果没有可抽取的，输出 []。"
    "只输出 JSON 数组，元素形如:\n"
    '  {"type":"mastery_update","concept_id":"loop","score":0.4,"evidence_note":"first time using forever"}\n'
    '  {"type":"error_observed","concept_id":"event","description":"把 when-flag-clicked 当成 forever 循环了"}\n'
    '  {"type":"preference_set","key":"topic","value":"猫和狗的动画","source":"self"}\n'
    "不要 markdown 包裹，不要解释。"
)


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_events(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and "type" in d]


def extract_node(state: TurnState) -> TurnState:
    if not state.get("student_utterance"):
        return {"extracted_events": []}
    log.info("[%s] extract... (parallel with respond)", state["student_id"])
    raw = MIMO.chat(
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": f"学生说: {state['student_utterance']}"},
        ],
        temperature=0.0,
        max_tokens=3072,
    )
    events = _parse_events(raw)
    log.info("extract: parsed %d event(s) from utterance", len(events))
    return {"extracted_events": events}


def save_node(state: TurnState) -> TurnState:
    sid = state["student_id"]
    log.info("[%s] save...", sid)
    saved = 0

    # 1) write LLM-extracted events
    for ev in state.get("extracted_events") or []:
        try:
            event_obj = S.EventEnvelope.model_validate({"event": ev}).event
            SERVICE.handle_event(sid, event_obj)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("skip invalid extracted event %r: %s", ev, exc)

    # 2) write harness-supplied events (deterministic stuff that doesn't
    #    survive a LLM round-trip well: session_end, project_saved, …)
    for ev in state.get("extra_events") or []:
        try:
            event_obj = S.EventEnvelope.model_validate({"event": ev}).event
            SERVICE.handle_event(sid, event_obj)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("skip invalid extra event %r: %s", ev, exc)

    # 3) write the turn pair as a dialog summary if we actually replied
    if state.get("student_utterance") and state.get("agent_reply"):
        SERVICE.handle_dialog(
            sid,
            [
                S.DialogTurn(role="student", content=state["student_utterance"]),
                S.DialogTurn(role="agent", content=state["agent_reply"]),
            ],
        )

    return {"saved_event_count": saved}


def build_graph():
    g = StateGraph(TurnState)
    g.add_node("recall", recall_node)
    g.add_node("respond", respond_node)
    g.add_node("extract", extract_node)
    g.add_node("save", save_node)
    g.add_edge(START, "recall")
    # Fan out: respond + extract are independent MiMo calls, run them
    # concurrently to roughly halve wall-clock per turn.
    g.add_edge("recall", "respond")
    g.add_edge("recall", "extract")
    # Fan in: save waits for both branches.
    g.add_edge("respond", "save")
    g.add_edge("extract", "save")
    g.add_edge("save", END)
    return g.compile()


# ---------------- scenario ----------------


def run_turn(
    graph,
    student_id: str,
    utterance: str | None,
    extra_events: list[dict] | None = None,
) -> TurnState:
    state: TurnState = {
        "student_id": student_id,
        "student_utterance": utterance or "",
        "extra_events": extra_events or [],
    }
    out = graph.invoke(state)
    if utterance:
        log.info("[%s] student> %s", student_id, utterance)
        log.info("[%s] agent  > %s", student_id, out.get("agent_reply", "")[:200])
    return out


def session_one(graph) -> None:
    sid = "alice"
    log.info("=== alice / session 1 ===")

    # First turn: preference declaration. Let the extractor pick it up.
    run_turn(graph, sid, "我喜欢做猫和狗的动画。")

    # Loops + forever: should produce mastery_update on `loop`.
    run_turn(graph, sid, "我想让小猫一直跑，是不是要用 forever 循环？")

    # Two event-related confusions in two turns: should trigger F cluster.
    run_turn(graph, sid, "我把 when-flag-clicked 当成 forever 循环用了，不太对。")
    run_turn(graph, sid, "我希望帽子积木一直触发，但好像不行。")

    # Save project + session end deterministically (don't rely on LLM
    # to invent project_id).
    run_turn(
        graph,
        sid,
        utterance=None,
        extra_events=[
            {
                "type": "project_saved",
                "project_id": "proj-1",
                "title": "跑步小猫",
                "structure": {"sprites": 1, "scripts": 2},
                "highlights": ["首次用 forever"],
                "issues": ["小猫飞出屏幕"],
            },
            {
                "type": "session_end",
                "current_project_id": "proj-1",
                "current_topic": "动画",
                "pending_task": "下次解决小猫飞出屏幕",
            },
        ],
    )


def session_two_and_assert(graph) -> None:
    sid = "alice"
    log.info("=== alice / session 2 (memory should kick in) ===")

    # Open the session with something topically adjacent.
    out = run_turn(graph, sid, "我又来了，今天继续做小猫吗？")

    profile = out["recalled_state"]
    session = profile["session"]
    log.info("recalled session: %s", session)
    log.info(
        "recalled mastery: %s",
        [(m["concept_id"], round(m["score"], 2)) for m in profile["mastery"]],
    )
    log.info(
        "recalled prefs:   %s",
        [(p["key"], p["value"]) for p in profile["preferences"]],
    )
    log.info(
        "recalled projects: %s",
        [(p["project_id"], p["title"]) for p in profile["projects"]],
    )
    log.info(
        "recalled misconcepts: %s",
        [(m["pattern_id"], m["occurrences"]) for m in profile["misconceptions"]],
    )

    # B: pending task survived
    assert session["current_project_id"] == "proj-1", session
    assert session["pending_task"] == "下次解决小猫飞出屏幕", session

    # D: project survived
    assert any(p["project_id"] == "proj-1" for p in profile["projects"])

    # E: preference survived (LLM-extracted in session 1)
    pref_keys = {p["key"] for p in profile["preferences"]}
    assert "topic" in pref_keys, f"preference 'topic' not extracted, got {pref_keys}"

    # A: mastery has at least 1 concept (LLM-extracted)
    assert profile["mastery"], "no mastery records — LLM extraction failed"

    # F: at least one misconception cluster from session 1's repeat errors
    assert profile["misconceptions"], "no misconceptions clustered"

    # C: dialog recall returns at least one hit grounded in session 1
    hits = out.get("recalled_dialog") or []
    assert hits, "dialog recall returned no hits"
    log.info("top dialog recall: score=%.3f summary=%s",
             hits[0]["score"], hits[0]["payload"].get("summary", "")[:80])


def assert_bob_isolation(graph) -> None:
    log.info("=== bob / isolation ===")
    out = run_turn(graph, "bob", "我刚学会用变量。")
    bob_profile = out["recalled_state"]
    assert bob_profile["projects"] == []
    assert bob_profile["preferences"] == []

    alice_state = SERVICE.get_state("alice")
    assert all(m.concept_id != "variable" for m in alice_state.mastery)
    log.info("alice unaffected by bob: ok")


def main() -> None:
    if not os.environ.get("MIMO_API_KEY"):
        raise SystemExit("MIMO_API_KEY missing — run via `uv run --env-file .env ...`")

    graph = build_graph()
    log.info("graph compiled, using MiMo model=%s", MIMO.model)

    session_one(graph)
    session_two_and_assert(graph)
    assert_bob_isolation(graph)

    log.info("=== LangGraph + MiMo + memory_module: all assertions passed ===")


if __name__ == "__main__":
    main()
