r"""Smoke test for the decoupled context-management feature.

Run from the memory module workspace:

    uv run python scripts/smoke_context.py
    # or: .\.venv\Scripts\python.exe scripts\smoke_context.py

Covers the service layer directly and the HTTP surface via TestClient,
including recency windowing, pinned-message retention, and clear/reset.
"""

from __future__ import annotations

import os
import tempfile
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
    ContextService,
    LLMSummarizingCompactor,
    SQLiteContextStore,
    build_default_context_service,
    build_default_user_context_service,
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


def smoke_context_summarizing() -> None:
    """LLM-summarizing compactor: overflow is summarized, not dropped.

    Uses a deterministic fake summarizer so no real LLM is called.
    """

    class _FakeSummarizer:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def summarize(self, messages):  # type: ignore[no-untyped-def]
            self.calls.append([m.content for m in messages])
            return "SUMMARY(" + ",".join(m.content for m in messages) + ")"

    fake = _FakeSummarizer()
    store = SQLiteContextStore(db_path=":memory:")
    svc = ContextService(
        store, LLMSummarizingCompactor(fake, summary_token_budget=10)
    )
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

    # Budget 50: reserve 10 for the summary -> eff 40; pinned(10) leaves 30,
    # which fits m3 only. m1+m2 overflow -> summarized into one synthetic msg.
    win = svc.get_window(sid, sess, max_tokens=50)
    assert win.truncated is True, win
    assert win.compaction_note and "2" in win.compaction_note, win
    assert [m.role for m in win.messages] == ["system", "system", "user"], win
    summary_msg = win.messages[1]
    assert summary_msg.metadata.get("synthetic") is True, summary_msg
    assert summary_msg.content == "SUMMARY(m1,m2)", summary_msg
    assert [m.content for m in win.messages] == ["sys", "SUMMARY(m1,m2)", "m3"], win
    assert win.total_tokens <= 50, win
    assert fake.calls == [["m1", "m2"]], fake.calls

    # No budget -> nothing overflows -> no summary, LLM never called again.
    win = svc.get_window(sid, sess)
    assert win.returned_messages == 4 and win.truncated is False, win
    assert win.compaction_note is None, win
    assert len(fake.calls) == 1, fake.calls


def smoke_context_isolation() -> None:
    """Cross-user isolation: two students sharing the SAME session_id must
    never see each other's context, and their seq counters are independent."""
    svc = build_default_context_service(db_path=":memory:")
    sess = "shared-session"  # identical session id for both users on purpose

    a = svc.append(
        "alice",
        sess,
        [
            ContextMessage(role="system", content="alice-sys", pinned=True),
            ContextMessage(role="user", content="alice-msg"),
        ],
    )
    b = svc.append("bob", sess, [ContextMessage(role="user", content="bob-msg")])
    assert a.total_messages == 2, a
    assert b.total_messages == 1, b  # bob's counts are independent of alice's

    # Windows never leak across users.
    aw = svc.get_window("alice", sess)
    bw = svc.get_window("bob", sess)
    assert [m.content for m in aw.messages] == ["alice-sys", "alice-msg"], aw
    assert [m.content for m in bw.messages] == ["bob-msg"], bw
    # seq is per-(student, session): bob starts at 1, not after alice's 2.
    assert [m.seq for m in bw.messages] == [1], bw

    # Stats are independent.
    assert svc.stats("alice", sess).message_count == 2
    assert svc.stats("bob", sess).message_count == 1

    # Clearing one user leaves the other intact.
    assert svc.clear("alice", sess) == 2
    assert svc.stats("alice", sess).message_count == 0
    assert svc.stats("bob", sess).message_count == 1, "bob wiped by alice's clear"
    assert [m.content for m in svc.get_window("bob", sess).messages] == ["bob-msg"]

    # HTTP layer: same session path segment, different students, no leakage.
    client = TestClient(build_app(db_path=":memory:"))
    for sid, content in (("u1", "u1-secret"), ("u2", "u2-secret")):
        resp = client.post(
            f"/students/{sid}/context/shared/messages",
            json={"messages": [{"role": "user", "content": content}]},
        )
        assert resp.status_code == 200, resp.text
    w1 = client.get("/students/u1/context/shared").json()
    w2 = client.get("/students/u2/context/shared").json()
    assert [m["content"] for m in w1["messages"]] == ["u1-secret"], w1
    assert [m["content"] for m in w2["messages"]] == ["u2-secret"], w2


def smoke_context_user() -> None:
    """Cross-session per-user context is a decoupled service keyed by the
    student alone (no session in its surface), isolated per user, reusing
    the same budget windowing."""
    user = build_default_user_context_service(db_path=":memory:")

    user.append(
        "alice",
        [
            ContextMessage(role="system", content="goal", pinned=True),
            ContextMessage(role="user", content="u1"),
        ],
    )
    win = user.get_window("alice")
    assert win.student_id == "alice", win
    assert win.session_id is None, win  # no session dimension exposed
    assert [m.content for m in win.messages] == ["goal", "u1"], win
    assert user.stats("alice").message_count == 2
    assert user.stats("alice").session_id is None

    # Isolated per user.
    user.append("bob", [ContextMessage(role="user", content="b1")])
    assert [m.content for m in user.get_window("bob").messages] == ["b1"]
    assert user.stats("alice").message_count == 2  # untouched by bob

    # Budget windowing on the user stream (pinned kept, oldest dropped).
    user.clear("alice")
    user.append(
        "alice",
        [
            ContextMessage(role="system", content="sys", pinned=True, token_count=10),
            ContextMessage(role="user", content="a", token_count=30),
            ContextMessage(role="user", content="b", token_count=30),
        ],
    )
    win = user.get_window("alice", max_tokens=45)  # pinned 10 + one 30 fits
    assert win.truncated is True, win
    assert [m.content for m in win.messages] == ["sys", "b"], win

    # HTTP: /user-context endpoints; session and user streams stay separate
    # within one app.
    client = TestClient(build_app(db_path=":memory:"))
    ubase = "/students/carol/user-context"
    assert (
        client.post(
            f"{ubase}/messages",
            json={"messages": [{"role": "user", "content": "c-user", "token_count": 5}]},
        ).status_code
        == 200
    )
    client.post(
        "/students/carol/context/sX/messages",
        json={"messages": [{"role": "user", "content": "c-session"}]},
    )
    uwin = client.get(ubase).json()
    assert [m["content"] for m in uwin["messages"]] == ["c-user"], uwin
    assert uwin["session_id"] is None, uwin
    assert client.get(f"{ubase}/stats").json()["message_count"] == 1
    assert client.get(ubase, params={"max_turns": 0}).status_code == 400  # guard reused
    # The per-session stream for the same student is independent.
    assert [
        m["content"] for m in client.get("/students/carol/context/sX").json()["messages"]
    ] == ["c-session"]
    assert client.delete(ubase).json()["deleted"] == 1
    assert client.get(ubase).json()["total_messages"] == 0
    # Deleting the user stream did not touch the session stream.
    assert client.get("/students/carol/context/sX").json()["total_messages"] == 1


def smoke_context_decoupled_tables() -> None:
    """Per-session and per-user streams live in independent tables in the
    SAME db file and never cross-talk; the table name is validated."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    sess = SQLiteContextStore(db_path=path, table="context_message")
    user = SQLiteContextStore(db_path=path, table="user_context_message")
    try:
        sess.append_messages(
            "alice", "sess-1", [ContextMessage(role="user", content="sess")]
        )
        user.append_messages("alice", "", [ContextMessage(role="user", content="user")])
        assert [m.content for m in sess.list_messages("alice", "sess-1")] == ["sess"]
        assert [m.content for m in user.list_messages("alice", "")] == ["user"]
        # Each table owns its own seq, both starting at 1.
        assert sess.list_messages("alice", "sess-1")[0].seq == 1
        assert user.list_messages("alice", "")[0].seq == 1
        # Clearing one table leaves the other intact.
        assert user.clear("alice", "") == 1
        assert sess.stats("alice", "sess-1").message_count == 1
    finally:
        sess.close()
        user.close()
        os.remove(path)

    # Invalid table names are rejected (defense against SQL interpolation).
    try:
        SQLiteContextStore(db_path=":memory:", table="bad-name")
        raise AssertionError("expected ValueError for bad table name")
    except ValueError:
        pass


def main() -> None:
    smoke_context_service()
    smoke_context_token_budget()
    smoke_context_summarizing()
    smoke_context_store_limit()
    smoke_context_isolation()
    smoke_context_user()
    smoke_context_decoupled_tables()
    smoke_context_http()
    print("context smoke ok")


if __name__ == "__main__":
    main()
