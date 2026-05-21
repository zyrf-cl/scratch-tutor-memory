r"""Smoke test for PracticeAgent memory persistence.

Run from the memory module workspace:

    .\.venv\Scripts\python.exe scripts\smoke_agent_memory.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parent.parent
AGENT_REPO = ROOT / "ai-programming-system-backend_hxh"
if str(AGENT_REPO) not in sys.path:
    sys.path.insert(0, str(AGENT_REPO))

os.environ.setdefault("MEMORY_MODULE_EMBEDDER", "hash")

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from agent.memory_client import AgentMemoryClient  # noqa: E402
from agent.practice_agent import classify_problem, update_memory  # noqa: E402
from agent.report import (  # noqa: E402
    AgentState,
    DiffReportData,
    EvaluationReport,
    EvaluationResultData,
)
from memory_module.service import build_default_service  # noqa: E402
from memory_module.api import build_app  # noqa: E402
from memory_module import schemas as S  # noqa: E402
from memory_module.extractor import MiMoAffectiveSignalDetector  # noqa: E402


def smoke_memory_module_service() -> None:
    svc = build_default_service(db_path=":memory:", use_mimo=False)

    for session_id, level in (("s1", 1), ("s2", 2)):
        svc.handle_agent_turn(
            "student-a",
            S.AgentTurnWrite(
                session_id=session_id,
                task_id="task-loop",
                project_name="proj",
                passed=False,
                score=0.5,
                primary_error_type="WRONG_STRUCTURE",
                concept_id="loop",
                feedback_level=level,
                feedback_text=f"hint {level}",
                user_text="我还是不会，这个好难",
                diff_summary="missing loop",
                highlight_node_ids=["node-loop"],
            ),
        )

    snap = svc.get_agent_memory_snapshot(
        "student-a",
        session_id="s3",
        task_id="task-loop",
    )
    assert snap.recent_feedback_level == 2, snap
    assert snap.recent_error_types[-2:] == [
        "WRONG_STRUCTURE",
        "WRONG_STRUCTURE",
    ], snap
    assert "loop" in snap.weak_concepts, snap

    snap = svc.get_agent_memory_snapshot(
        "student-a",
        session_id="s3",
        task_id="task-loop",
    )
    assert snap.affective_state["frustration"] > 0.0, snap
    assert snap.affective_state["confusion"] > 0.0, snap


def smoke_practice_agent_memory_path() -> None:
    client = AgentMemoryClient(db_path=":memory:", project_root=AGENT_REPO)

    state1 = AgentState(
        task_id="task-loop",
        user_id="student-b",
        session_id="session-1",
        project_name="proj",
    )
    state1.memory_snapshot = client.load_snapshot(
        user_id=state1.user_id,
        session_id=state1.session_id,
        task_id=state1.task_id,
    )
    state1.evaluation_result = EvaluationResultData(status="fail")
    state1.evaluation_report = EvaluationReport(
        project_name="proj",
        passed=False,
        score=0.5,
    )
    state1.diff_report = DiffReportData(
        error_types=["WRONG_STRUCTURE"],
        diff_summary="missing loop",
        node_ids=["node-loop"],
        equivalence_score=0.5,
    )
    state1 = classify_problem(state1)
    assert state1.feedback_level == 1, state1.to_dict()
    state1.user_text = "我不太懂"
    state1.decision_result.student_feedback = "first hint"
    client.persist_state(update_memory(state1))

    state2 = AgentState(
        task_id="task-loop",
        user_id="student-b",
        session_id="session-2",
        project_name="proj",
    )
    state2.memory_snapshot = client.load_snapshot(
        user_id=state2.user_id,
        session_id=state2.session_id,
        task_id=state2.task_id,
    )
    state2.evaluation_result = EvaluationResultData(status="fail")
    state2.evaluation_report = EvaluationReport(
        project_name="proj",
        passed=False,
        score=0.4,
    )
    state2.diff_report = DiffReportData(
        error_types=["WRONG_STRUCTURE"],
        diff_summary="still missing loop",
        node_ids=["node-loop"],
        equivalence_score=0.4,
    )
    state2 = classify_problem(state2)
    assert state2.feedback_level == 2, state2.to_dict()

    state2.user_text = "还是不会，卡住了"
    state2.decision_result.student_feedback = "second hint"
    client.persist_state(update_memory(state2))
    loaded = client.load_snapshot(
        user_id="student-b",
        session_id="session-3",
        task_id="task-loop",
    )
    assert loaded.affective_state["frustration"] > 0.0, loaded.to_dict()
    assert loaded.affective_state["confusion"] > 0.0, loaded.to_dict()


def smoke_http_agent_memory_path() -> None:
    app = build_app(db_path=":memory:")
    client = TestClient(app)

    resp = client.post(
        "/students/student-c/agent-turns",
        json={
            "session_id": "http-s1",
            "task_id": "task-loop",
            "project_name": "proj",
            "passed": False,
            "score": 0.3,
            "primary_error_type": "WRONG_STRUCTURE",
            "concept_id": "loop",
            "feedback_level": 2,
            "feedback_text": "hint over http",
            "user_text": "这个太难了，我不会",
            "diff_summary": "missing forever block",
            "highlight_node_ids": ["node-loop"],
            "need_rag": False,
            "need_repair_validation": False,
        },
    )
    assert resp.status_code == 200, resp.text

    snap = client.get(
        "/students/student-c/agent-memory",
        params={"session_id": "http-s2", "task_id": "task-loop"},
    )
    assert snap.status_code == 200, snap.text
    payload = snap.json()
    assert payload["affective_state"]["frustration"] > 0.0, payload
    assert payload["affective_state"]["confusion"] > 0.0, payload


def smoke_llm_affective_service_path() -> None:
    if not os.environ.get("MIMO_API_KEY"):
        print("skip llm affective smoke: no MIMO_API_KEY")
        return

    svc = build_default_service(db_path=":memory:", use_mimo=False)
    svc._affective = MiMoAffectiveSignalDetector()  # type: ignore[attr-defined]
    svc.handle_agent_turn(
        "student-d",
        S.AgentTurnWrite(
            session_id="llm-s1",
            task_id="task-loop",
            project_name="proj",
            passed=False,
            score=0.2,
            primary_error_type="WRONG_STRUCTURE",
            concept_id="loop",
            feedback_level=2,
            feedback_text="hint",
            user_text="这个太难了，我完全不会",
            diff_summary="missing forever block",
            highlight_node_ids=["node-loop"],
        ),
    )
    events = svc._store.list_affective_events("student-d", task_id="task-loop", limit=5)  # type: ignore[attr-defined]
    assert events, "no affective events persisted"
    event = events[-1]
    assert event.frustration_score > 0.0, event
    assert event.confusion_score > 0.0, event
    assert event.evidence, event
    assert not event.evidence.startswith("mode=rule"), event


def main() -> None:
    smoke_memory_module_service()
    smoke_practice_agent_memory_path()
    smoke_http_agent_memory_path()
    smoke_llm_affective_service_path()
    print("agent memory smoke ok")


if __name__ == "__main__":
    main()
