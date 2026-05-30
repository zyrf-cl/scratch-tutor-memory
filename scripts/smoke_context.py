r"""Smoke test for the decoupled context-management feature.

Run from the memory module workspace:

    uv run python scripts/smoke_context.py
    # or: .\.venv\Scripts\python.exe scripts\smoke_context.py

Covers the service layer directly and the HTTP surface via TestClient,
including recency windowing, pinned-message retention, and clear/reset.
"""

from __future__ import annotations

import os
from pathlib import Path

# Keep the embedder cheap: build_app constructs the full MemoryService
# (which builds an embedder) at import time. Must be set BEFORE importing
# memory_module.api, exactly like scripts/smoke_agent_memory.py.
os.environ.setdefault("MEMORY_MODULE_EMBEDDER", "hash")

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from fastapi.testclient import TestClient  # noqa: E402

from memory_module.api import build_app  # noqa: E402
from memory_module.context import (  # noqa: E402
    ContextMessage,
    SQLiteContextStore,
    build_default_context_service,
)


def smoke_context_service() -> None:
    svc = build_default_context_service(db_path=":memory:")
    sid, sess = "student-ctx", "sess-1"

    res = svc.append(
        sid,
        sess,
        [
            ContextMessage(
                role="system", content="You are a Scratch tutor.", pinned=True
            ),
            ContextMessage(role="user", content="how do I make a sprite jump?"),
            ContextMessage(role="assistant", content="Use a change-y-by block."),
        ],
    )
    assert res.appended == 3, res
    assert res.total_messages == 3, res
    assert res.total_tokens > 0, res

    # No budget -> everything, chronological, not truncated.
    win = svc.get_window(sid, sess)
    assert win.returned_messages == 3, win
    assert win.total_messages == 3, win
    assert win.truncated is False, win
    assert [m.role for m in win.messages] == ["system", "user", "assistant"], win

    # max_turns counts the pinned system msg: 2 -> system + newest non-pinned.
    win = svc.get_window(sid, sess, max_turns=2)
    assert win.returned_messages == 2, win
    assert win.truncated is True, win
    roles = [m.role for m in win.messages]
    assert roles[0] == "system", roles            # pinned kept
    assert roles[-1] == "assistant", roles        # most recent kept
    assert "user" not in roles, roles             # oldest non-pinned dropped

    st = svc.stats(sid, sess)
    assert st.message_count == 3, st
    assert st.last_updated is not None, st

    deleted = svc.clear(sid, sess)
    assert deleted == 3, deleted
    assert svc.stats(sid, sess).message_count == 0
    empty = svc.get_window(sid, sess)
    assert empty.returned_messages == 0 and empty.total_messages == 0, empty


def smoke_context_token_budget() -> None:
    """Deterministic token windowing using explicit token_count values."""
    svc = build_default_context_service(db_path=":memory:")
    sid, sess = "s", "t"
    svc.append(
        sid,
        sess,
        [
            ContextMessage(role="system", content="sys", pinned=True, token_count=10),
            ContextMessage(role="user", content="m1", token_count=30),
            ContextMessage(role="user", content="m2", token_count=30),
            ContextMessage(role="user", content="m3", token_count=30),
        ],
    )

    # Budget 50: pinned eats 10, remaining 40 fits exactly one 30-token msg.
    win = svc.get_window(sid, sess, max_tokens=50)
    assert win.truncated is True, win
    assert [m.content for m in win.messages] == ["sys", "m3"], win
    assert win.total_tokens == 40, win

    # Pinned alone (10) exceeds a tiny budget: pinned is inviolable, all
    # non-pinned dropped, still flagged truncated.
    win = svc.get_window(sid, sess, max_tokens=5)
    assert [m.content for m in win.messages] == ["sys"], win
    assert win.truncated is True, win
    assert win.total_tokens == 10, win


def smoke_context_http() -> None:
    app = build_app(db_path=":memory:")
    client = TestClient(app)
    base = "/students/stu-http/context/sess-http"

    assert client.get("/health").json() == {"status": "ok"}

    resp = client.post(
        f"{base}/messages",
        json={
            "messages": [
                {"role": "system", "content": "sys prompt", "pinned": True, "token_count": 5},
                {"role": "user", "content": "hello", "token_count": 20},
                {"role": "assistant", "content": "hi there", "token_count": 20},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["appended"] == 3 and body["total_messages"] == 3, body

    resp = client.get(base)
    assert resp.status_code == 200, resp.text
    win = resp.json()
    assert win["returned_messages"] == 3 and win["truncated"] is False, win

    # Budget 25: pinned(5) + remaining 20 -> keep only the newest 20-token msg.
    resp = client.get(base, params={"max_tokens": 25})
    win = resp.json()
    assert win["truncated"] is True, win
    assert [m["content"] for m in win["messages"]] == ["sys prompt", "hi there"], win

    resp = client.get(base, params={"max_turns": 2})
    assert resp.json()["returned_messages"] == 2, resp.text

    resp = client.get(f"{base}/stats")
    assert resp.status_code == 200, resp.text
    assert resp.json()["message_count"] == 3, resp.text

    # Guards + empty-body rejection.
    assert client.get(base, params={"max_turns": 0}).status_code == 400
    assert client.post(f"{base}/messages", json={"messages": []}).status_code == 400

    resp = client.delete(base)
    assert resp.status_code == 200 and resp.json()["deleted"] == 3, resp.text
    assert client.get(base).json()["total_messages"] == 0


def smoke_context_store_limit() -> None:
    """list_messages limit semantics — guards the limit=0 slice footgun."""
    store = SQLiteContextStore(db_path=":memory:")
    store.append_messages(
        "s", "t", [ContextMessage(role="user", content=f"m{i}") for i in range(5)]
    )
    assert len(store.list_messages("s", "t")) == 5
    assert [m.content for m in store.list_messages("s", "t", limit=2)] == ["m3", "m4"]
    # limit=0 must return nothing, NOT the whole list (records[-0:] footgun).
    assert store.list_messages("s", "t", limit=0) == []


def main() -> None:
    smoke_context_service()
    smoke_context_token_budget()
    smoke_context_store_limit()
    smoke_context_http()
    print("context smoke ok")


if __name__ == "__main__":
    main()
