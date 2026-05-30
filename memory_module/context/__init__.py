"""Decoupled context-management feature for the memory module.

This subpackage manages *short-term working context* — the running
conversation an agent pushes in and reads back a bounded window of to
feed its LLM. It is fully self-contained: it owns its own schemas,
SQLite store, compaction policy, service, and HTTP router, and shares
no state with the long-term :class:`~memory_module.service.MemoryService`.

Wiring it into an app is a one-liner at the composition root::

    from memory_module.context import (
        build_default_context_service,
        build_context_router,
    )

    app.include_router(build_context_router(build_default_context_service(db_path)))

See ``README.md`` ("上下文管理（独立功能）") for the design rationale and
``INTEGRATION.md`` for the HTTP contract.
"""

from __future__ import annotations

from .api import build_context_router
from .compaction import Compactor, CompactionResult, TruncatingCompactor
from .schemas import (
    AppendContextRequest,
    AppendResult,
    ContextMessage,
    ContextMessageRecord,
    ContextStats,
    ContextWindow,
    Role,
    estimate_tokens,
)
from .service import ContextService, build_default_context_service
from .store import CONTEXT_SCHEMA_SQL, ContextStore, SQLiteContextStore

__all__ = [
    # service + wiring
    "ContextService",
    "build_default_context_service",
    "build_context_router",
    # store
    "ContextStore",
    "SQLiteContextStore",
    "CONTEXT_SCHEMA_SQL",
    # compaction
    "Compactor",
    "CompactionResult",
    "TruncatingCompactor",
    # schemas
    "Role",
    "estimate_tokens",
    "ContextMessage",
    "AppendContextRequest",
    "ContextMessageRecord",
    "ContextWindow",
    "AppendResult",
    "ContextStats",
]
