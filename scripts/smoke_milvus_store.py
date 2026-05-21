"""Smoke-test MilvusStore against a running Milvus service.

Run:

    uv sync --extra milvus
    MEMORY_MODULE_STORE=milvus MEMORY_MODULE_EMBEDDER=hash uv run python scripts/smoke_milvus_store.py

The script uses a unique collection prefix by default so it does not touch
normal demo data. Set MEMORY_MODULE_MILVUS_PREFIX explicitly to reuse one.
"""

from __future__ import annotations

import os
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MEMORY_MODULE_STORE", "milvus")
os.environ.setdefault("MEMORY_MODULE_EMBEDDER", "hash")
os.environ.setdefault("MEMORY_MODULE_MILVUS_PREFIX", f"memory_module_smoke_{uuid.uuid4().hex[:8]}")

from memory_module import schemas as S  # noqa: E402
from memory_module.service import build_default_service  # noqa: E402


def main() -> None:
    svc = build_default_service(use_mimo=False)
    sid = "milvus-alice"

    svc.handle_event(sid, S.PreferenceSetEvent(key="topic", value="猫和狗的动画"))
    svc.handle_event(sid, S.MasteryUpdateEvent(concept_id="loop", score=0.6))
    svc.handle_event(
        sid,
        S.ProjectSavedEvent(
            project_id="proj-1",
            title="跑步小猫",
            structure={"sprites": 1},
            highlights=["首次用 forever"],
            issues=["小猫飞出屏幕"],
        ),
    )
    svc.handle_event(
        sid,
        S.ErrorObservedEvent(
            concept_id="event",
            description="把 when-flag-clicked 当成 forever 循环了",
        ),
    )
    svc.handle_event(
        sid,
        S.ErrorObservedEvent(
            concept_id="event",
            description="希望帽子积木一直触发",
        ),
    )
    svc.handle_dialog(
        sid,
        [
            S.DialogTurn(role="student", content="我想让小猫一直跑"),
            S.DialogTurn(role="agent", content="可以用 forever 循环"),
        ],
    )
    svc.handle_event(
        sid,
        S.SessionEndEvent(
            current_project_id="proj-1",
            current_topic="动画",
            pending_task="下次解决小猫飞出屏幕",
        ),
    )

    state = svc.get_state(sid)
    assert state.session.pending_task == "下次解决小猫飞出屏幕", state
    assert state.mastery and state.mastery[0].concept_id == "loop", state
    assert state.projects and state.projects[0].project_id == "proj-1", state
    assert state.preferences and state.preferences[0].value == "猫和狗的动画", state
    assert state.misconceptions and state.misconceptions[0].pattern_id == "concept:event", state
    assert svc.recall(sid, "小猫一直跑", kind="dialog", top_k=1), "dialog recall empty"
    assert svc.recall(sid, "event", kind="misconception", top_k=1), "misconception recall empty"

    for session_id, level in (("s1", 1), ("s2", 2)):
        svc.handle_agent_turn(
            sid,
            S.AgentTurnWrite(
                session_id=session_id,
                task_id="loop-cat",
                passed=False,
                score=0.4,
                primary_error_type="WRONG_STRUCTURE",
                concept_id="loop",
                feedback_level=level,
                feedback_text=f"hint {level}",
                diff_summary="missing forever block",
            ),
        )
    svc.handle_affective_event(
        sid,
        S.AffectiveEventWrite(
            session_id="s2",
            task_id="loop-cat",
            frustration_score=0.8,
            confusion_score=0.7,
            confidence_score=0.2,
        ),
    )
    snap = svc.get_agent_memory_snapshot(sid, session_id="s3", task_id="loop-cat")
    assert snap.recent_feedback_level == 2, snap
    assert "loop" in snap.weak_concepts, snap
    assert snap.affective_state["frustration"] == 0.8, snap

    print("milvus store smoke ok")


if __name__ == "__main__":
    main()
