"""FastAPI router for the context-management feature.

Thin routes, same as the parent ``api.py``: validate, delegate to
:class:`ContextService`, return. The router is built around an already
constructed service (captured via a closure dependency) so it can be
mounted onto any app — or served standalone — without global state.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from . import schemas as CS
from .service import ContextService

logger = logging.getLogger(__name__)

# Light sanity caps, mirroring the parent's top_k guard. Generous on
# purpose — they exist to reject obviously bad input, not to enforce
# policy (the real budget is the compactor's job).
_MAX_TURNS_CAP = 10_000
_MAX_TOKENS_CAP = 2_000_000


def build_context_router(service: ContextService) -> APIRouter:
    router = APIRouter(tags=["context"])

    def _get_service() -> ContextService:
        return service

    @router.post(
        "/students/{student_id}/context/{session_id}/messages",
        response_model=CS.AppendResult,
    )
    def append_messages(
        student_id: str,
        session_id: str,
        request: CS.AppendContextRequest,
        svc: ContextService = Depends(_get_service),
    ) -> CS.AppendResult:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        return svc.append(student_id, session_id, request.messages)

    @router.get(
        "/students/{student_id}/context/{session_id}",
        response_model=CS.ContextWindow,
    )
    def get_window(
        student_id: str,
        session_id: str,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        svc: ContextService = Depends(_get_service),
    ) -> CS.ContextWindow:
        if max_tokens is not None and not (1 <= max_tokens <= _MAX_TOKENS_CAP):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be in [1, {_MAX_TOKENS_CAP}]",
            )
        if max_turns is not None and not (1 <= max_turns <= _MAX_TURNS_CAP):
            raise HTTPException(
                status_code=400,
                detail=f"max_turns must be in [1, {_MAX_TURNS_CAP}]",
            )
        return svc.get_window(
            student_id,
            session_id,
            max_tokens=max_tokens,
            max_turns=max_turns,
        )

    @router.get(
        "/students/{student_id}/context/{session_id}/stats",
        response_model=CS.ContextStats,
    )
    def get_stats(
        student_id: str,
        session_id: str,
        svc: ContextService = Depends(_get_service),
    ) -> CS.ContextStats:
        return svc.stats(student_id, session_id)

    @router.delete("/students/{student_id}/context/{session_id}")
    def clear_context(
        student_id: str,
        session_id: str,
        svc: ContextService = Depends(_get_service),
    ) -> dict[str, int]:
        return {"deleted": svc.clear(student_id, session_id)}

    return router


__all__ = ["build_context_router"]
