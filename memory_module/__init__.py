"""Memory module for a Scratch teaching agent.

Public entry points:

* :mod:`memory_module.api` — FastAPI app (``memory_module.api:app``)
* :mod:`memory_module.service` — :class:`MemoryService` orchestrator
* :mod:`memory_module.schemas` — Pydantic request/response models

See ``README.md`` for the full design overview.
"""

from .service import MemoryService, build_default_service

__all__ = ["MemoryService", "build_default_service"]
