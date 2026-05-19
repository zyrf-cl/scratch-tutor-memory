"""End-to-end demo for the memory module.

Simulates a Scratch-teaching scenario:

* Student Alice goes through session 1 (mastery, errors, dialog,
  project, preference, then session_end).
* "Session 2" — the agent asks the memory module what it remembers
  about Alice. We assert all six memory types contain the expected
  evidence.
* Student Bob is added to verify per-student isolation.

Runs entirely in-process via FastAPI's TestClient against an in-memory
SQLite database — no uvicorn needed.

::

    uv run python demo.py
"""

from __future__ import annotations

import logging
import sys

# Force UTF-8 stdout so Chinese strings render correctly on Windows
# consoles that default to cp936/gbk.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from memory_module.api import build_app  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("demo")


def _post(client: TestClient, path: str, payload: dict) -> dict:
    resp = client.post(path, json=payload)
    assert resp.status_code == 200, f"POST {path} -> {resp.status_code}: {resp.text}"
    return resp.json()


def _get(client: TestClient, path: str, **params) -> dict | list:
    resp = client.get(path, params=params)
    assert resp.status_code == 200, f"GET {path} -> {resp.status_code}: {resp.text}"
    return resp.json()


def alice_session_one(client: TestClient) -> None:
    sid = "alice"

    # E preference (declared up front)
    _post(
        client,
        f"/students/{sid}/events",
        {"event": {"type": "preference_set", "key": "topic", "value": "猫和狗的动画", "source": "self"}},
    )

    # A mastery: tries loops + events
    _post(
        client,
        f"/students/{sid}/events",
        {"event": {"type": "mastery_update", "concept_id": "loop", "score": 0.3, "evidence_note": "first attempt"}},
    )
    _post(
        client,
        f"/students/{sid}/events",
        {"event": {"type": "mastery_update", "concept_id": "event", "score": 0.5}},
    )

    # F errors (two on 'event' → should promote to a misconception)
    _post(
        client,
        f"/students/{sid}/events",
        {"event": {"type": "error_observed", "concept_id": "event", "description": "把 when-flag-clicked 当成 forever 循环了"}},
    )
    _post(
        client,
        f"/students/{sid}/events",
        {"event": {"type": "error_observed", "concept_id": "event", "description": "希望帽子积木一直触发"}},
    )

    # C dialog
    _post(
        client,
        f"/students/{sid}/dialog",
        {
            "turns": [
                {"role": "agent", "content": "今天我们让小猫动起来"},
                {"role": "student", "content": "我想让小猫一直跑"},
                {"role": "agent", "content": "可以用 forever 循环包住移动积木"},
                {"role": "student", "content": "那要怎么让它撞到边再回来"},
            ]
        },
    )

    # D project saved
    _post(
        client,
        f"/students/{sid}/events",
        {
            "event": {
                "type": "project_saved",
                "project_id": "proj-1",
                "title": "跑步小猫",
                "structure": {"sprites": 1, "scripts": 2},
                "highlights": ["首次用 forever"],
                "issues": ["小猫飞出屏幕"],
            }
        },
    )

    # B session end
    _post(
        client,
        f"/students/{sid}/events",
        {
            "event": {
                "type": "session_end",
                "current_project_id": "proj-1",
                "current_topic": "动画",
                "pending_task": "下次解决小猫飞出屏幕",
            }
        },
    )


def assert_alice_session_two_recall(client: TestClient) -> None:
    sid = "alice"
    state = _get(client, f"/students/{sid}/state")
    log.info("alice.state.session     = %s", state["session"])
    log.info("alice.state.mastery     = %s", [(m["concept_id"], m["score"]) for m in state["mastery"]])
    log.info("alice.state.preferences = %s", [(p["key"], p["value"]) for p in state["preferences"]])
    log.info("alice.state.projects    = %s", [(p["project_id"], p["title"]) for p in state["projects"]])
    log.info("alice.state.misconcept. = %s", [(m["pattern_id"], m["occurrences"]) for m in state["misconceptions"]])

    # B continuity
    assert state["session"]["current_project_id"] == "proj-1"
    assert state["session"]["pending_task"] == "下次解决小猫飞出屏幕"

    # A mastery
    concepts = {m["concept_id"]: m for m in state["mastery"]}
    assert "loop" in concepts and abs(concepts["loop"]["score"] - 0.3) < 1e-6
    assert "event" in concepts and abs(concepts["event"]["score"] - 0.5) < 1e-6

    # E preferences
    assert any(p["key"] == "topic" and p["value"] == "猫和狗的动画" for p in state["preferences"])

    # D projects
    assert len(state["projects"]) == 1 and state["projects"][0]["project_id"] == "proj-1"

    # F misconceptions: two errors on `event` → promoted
    misc = state["misconceptions"]
    assert len(misc) == 1 and misc[0]["pattern_id"] == "concept:event"
    assert misc[0]["occurrences"] >= 1

    # C dialog recall: should pull back the small-cat session for the
    # right query.
    recall = _get(client, f"/students/{sid}/recall", query="小猫一直跑", kind="dialog", top_k=3)
    assert isinstance(recall, list) and len(recall) >= 1
    log.info("alice.recall(小猫一直跑) -> top hit summary='%s' score=%.3f",
             recall[0]["payload"]["summary"], recall[0]["score"])

    # F misconception recall: query should find the event-related cluster.
    recall_m = _get(client, f"/students/{sid}/recall", query="event", kind="misconception", top_k=3)
    assert isinstance(recall_m, list) and len(recall_m) >= 1
    log.info("alice.recall(event)/misconcept -> %s", recall_m[0]["payload"]["pattern_id"])


def assert_bob_isolation(client: TestClient) -> None:
    bob_state_before = _get(client, "/students/bob/state")
    assert bob_state_before["mastery"] == []
    assert bob_state_before["projects"] == []
    assert bob_state_before["preferences"] == []
    assert bob_state_before["misconceptions"] == []
    log.info("bob has clean slate before any writes: ok")

    _post(
        client,
        "/students/bob/events",
        {"event": {"type": "mastery_update", "concept_id": "variable", "score": 0.8}},
    )

    # Cross-check: alice should not have learned 'variable'
    alice_state = _get(client, "/students/alice/state")
    alice_concepts = {m["concept_id"] for m in alice_state["mastery"]}
    assert "variable" not in alice_concepts, "Alice's mastery polluted by Bob"
    log.info("alice unaffected by bob's writes: ok")


def main() -> None:
    log.info("=== building app with in-memory SQLite ===")
    app = build_app(db_path=":memory:")
    client = TestClient(app)

    log.info("=== alice: session 1 (writes) ===")
    alice_session_one(client)

    log.info("=== alice: session 2 (recall + assertions) ===")
    assert_alice_session_two_recall(client)

    log.info("=== bob: isolation check ===")
    assert_bob_isolation(client)

    log.info("=== demo OK: all 6 memory types verified ===")


if __name__ == "__main__":
    main()
