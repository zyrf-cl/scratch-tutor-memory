"""Thin MiMo (Xiaomi) OpenAI-compatible chat client.

Reads ``MIMO_API_KEY`` / ``MIMO_BASE_URL`` / ``MIMO_MODEL`` from the
environment (typically loaded from ``.env``). Exposes a single
:func:`chat` helper that returns the final assistant ``content``,
discarding the model's ``reasoning_content``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro"


class MiMoClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not self._api_key:
            raise RuntimeError("MIMO_API_KEY is not set")
        self._base_url = (base_url or os.environ.get("MIMO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or os.environ.get("MIMO_MODEL") or DEFAULT_MODEL
        self._timeout = timeout
        self._client = httpx.Client(timeout=self._timeout)

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 3072,
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
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        usage = data.get("usage", {})
        if not content:
            # Model used the entire budget thinking. Take the tail of
            # reasoning as a last resort — it's usually the conclusion.
            rc = (msg.get("reasoning_content") or "").strip()
            logger.warning(
                "mimo.chat empty content (model=%s, completion=%s, reasoning=%s); "
                "consider raising max_tokens",
                payload["model"],
                usage.get("completion_tokens"),
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            )
            content = rc[-400:] if rc else ""
        logger.debug(
            "mimo.chat model=%s tokens=%s/%s",
            payload["model"],
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        return content


__all__ = ["MiMoClient", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]
