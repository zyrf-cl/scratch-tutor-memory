"""Thin Kimi (Moonshot AI) OpenAI-compatible chat client.

Mirrors :class:`~memory_module.mimo_client.MiMoClient`. Reads
``KIMI_API_KEY`` (or ``MOONSHOT_API_KEY``) / ``KIMI_BASE_URL`` /
``KIMI_MODEL`` from the environment and exposes a single :func:`chat`
helper returning the assistant ``content``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# China endpoint by default; international accounts use api.moonshot.ai.
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "moonshot-v1-8k"


class KimiClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError("KIMI_API_KEY (or MOONSHOT_API_KEY) is not set")
        self._base_url = (
            base_url or os.environ.get("KIMI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self._model = model or os.environ.get("KIMI_MODEL") or DEFAULT_MODEL
        self._client = httpx.Client(timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        logger.debug(
            "kimi.chat model=%s tokens=%s",
            payload["model"],
            data.get("usage", {}).get("total_tokens"),
        )
        return content


__all__ = ["KimiClient", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]
