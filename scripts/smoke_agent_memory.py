r"""Smoke test for PracticeAgent memory persistence.

Run from the memory module workspace:

    .\.venv\Scripts\python.exe scripts\smoke_agent_memory.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AGENT_REPO = ROOT / "ai-programming-system-backend_hxh"
if str(AGENT_REPO) not in sys.path:
    sys.path.insert(0, str(AGENT_REPO))

os.environ.setdefault("MEMORY_MODULE_EMBEDDER", "hash")

from agent.memory_client import AgentMemoryClient  # noqa: E402
from agent.practice_agent import classify_problem, update_memory  # noqa: E402
from agent.report import (  # noqa: E402
    AgentState,
    DiffReportData,
    EvaluationReport,
    EvaluationResultData,
)
from memory_module.service import build_default_service  # noqa: E402
from memory_module import schemas as S  # noqa: E402


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

    svc.handle_affective_event(
        "student-a",
        S.AffectiveEventWrite(
            session_id="s2",
            task_id="task-loop",
            frustration_score=0.8,
            confidence_score=0.2,
            engagement_score=0.6,
            confusion_score=0.75,
            evidence="test affective signal",
        ),
    )
    snap = svc.get_agent_memory_snapshot(
        "student-a",
        session_id="s3",
        task_id="task-loop",
    )
    assert snap.affective_state["frustration"] == 0.8, snap
    assert snap.affective_state["confusion"] == 0.75, snap


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

    state2.decision_result.student_feedback = "second hint"
    client.persist_state(update_memory(state2))
    loaded = client.load_snapshot(
        user_id="student-b",
        session_id="session-3",
        task_id="task-loop",
    )
    assert loaded.affective_state["frustration"] > 0.0, loaded.to_dict()
    assert loaded.affective_state["confusion"] > 0.0, loaded.to_dict()


def main() -> None:
    smoke_memory_module_service()
    smoke_practice_agent_memory_path()
    print("agent memory smoke ok")


if __name__ == "__main__":
    main()
