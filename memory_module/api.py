"""FastAPI app exposing the memory module over HTTP.

Routes are intentionally thin: they validate the request, hand it
off to :class:`MemoryService`, and return the response. All policy
lives in the service layer.

Usage::

    uv run uvicorn memory_module.api:app --reload

For the demo we also expose :func:`build_app` so an in-process
:class:`fastapi.testclient.TestClient` can spin up a fresh service
backed by an in-memory SQLite database.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException

from . import schemas as S
from .service import MemoryService, build_default_service

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH_ENV = "MEMORY_MODULE_DB"


def build_app(db_path: str | None = None) -> FastAPI:
    """Construct a fresh FastAPI app with its own MemoryService.

    Pass ``":memory:"`` for an isolated in-process store (used by the
    demo). Pass a filesystem path for a persistent store.
    """
    if db_path is None:
        db_path = os.environ.get(DEFAULT_DB_PATH_ENV, ":memory:")

    service = build_default_service(db_path=db_path)
    application = FastAPI(title="Scratch Teaching Agent — Memory Module")

    def _get_service() -> MemoryService:
        return service

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/students/{student_id}/events")
    def post_event(
        student_id: str,
        envelope: S.EventEnvelope,
        svc: MemoryService = Depends(_get_service),
    ) -> dict[str, str]:
        try:
            svc.handle_event(student_id, envelope.event)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "ok"}

    @application.post("/students/{student_id}/dialog")
    def post_dialog(
        student_id: str,
        batch: S.DialogBatch,
        svc: MemoryService = Depends(_get_service),
    ) -> dict[str, int]:
        turn_id = svc.handle_dialog(student_id, batch.turns)
        return {"turn_id": turn_id}

    @application.get("/students/{student_id}/state", response_model=S.StudentState)
    def get_state(
        student_id: str,
        svc: MemoryService = Depends(_get_service),
    ) -> S.StudentState:
        return svc.get_state(student_id)

    @application.get(
        "/students/{student_id}/recall", response_model=list[S.RecallHit]
    )
    def get_recall(
        student_id: str,
        query: str,
        kind: Literal["dialog", "misconception"] = "dialog",
        top_k: int = 5,
        svc: MemoryService = Depends(_get_service),
    ) -> list[S.RecallHit]:
        if top_k <= 0 or top_k > 50:
            raise HTTPException(
                status_code=400, detail="top_k must be in [1, 50]"
            )
        try:
            return svc.recall(student_id, query=query, kind=kind, top_k=top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @application.post("/students/{student_id}/agent-turns")
    def post_agent_turn(
        student_id: str,
        turn: S.AgentTurnWrite,
        svc: MemoryService = Depends(_get_service),
    ) -> dict[str, int]:
        try:
            turn_id = svc.handle_agent_turn(student_id, turn)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"turn_id": turn_id}

    @application.post("/students/{student_id}/affective-events")
    def post_affective_event(
        student_id: str,
        event: S.AffectiveEventWrite,
        svc: MemoryService = Depends(_get_service),
    ) -> dict[str, int]:
        event_id = svc.handle_affective_event(student_id, event)
        return {"event_id": event_id}

    @application.get(
        "/students/{student_id}/agent-memory",
        response_model=S.AgentMemorySnapshot,
    )
    def get_agent_memory(
        student_id: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        svc: MemoryService = Depends(_get_service),
    ) -> S.AgentMemorySnapshot:
        return svc.get_agent_memory_snapshot(
            student_id,
            session_id=session_id,
            task_id=task_id,
        )

    return application


# Module-level app for `uvicorn memory_module.api:app`
app = build_app()


__all__ = ["app", "build_app"]
