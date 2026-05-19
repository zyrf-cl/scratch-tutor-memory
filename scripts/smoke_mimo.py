"""Smoke-test the MiMo chat endpoint over the OpenAI-compatible API."""

from __future__ import annotations

import os
import sys

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.environ.get("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
KEY = os.environ["MIMO_API_KEY"]
MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")


def main() -> None:
    resp = httpx.post(
        f"{BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个简洁的中文助手。只输出最终结论，不要解释思考过程。",
                },
                {
                    "role": "user",
                    "content": "用一句话总结这段对话：学生说想让小猫一直跑，又问怎么让它撞到边再回来。",
                },
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    print("=== content ===")
    print(msg.get("content"))
    print("=== reasoning (truncated) ===")
    rc = msg.get("reasoning_content") or ""
    print(rc[:200] + ("..." if len(rc) > 200 else ""))
    print("=== usage ===")
    print(data.get("usage"))


if __name__ == "__main__":
    main()
