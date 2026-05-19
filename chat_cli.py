"""Interactive REPL for the MiMo + LangGraph memory pipeline.

Persistence
-----------
Memory is written to a real SQLite file at ``./memory.db`` (overridable
via ``MEMORY_MODULE_DB``), so you can quit, come back tomorrow, and the
agent will still remember everyone.

Slash commands
--------------
==================  =================================================
``/help``           show this list
``/who``            list students that have any memory written
``/switch <sid>``   switch active student (creates one if new)
``/state``          dump full StudentState for active student
``/recall <q>``     semantic recall over dialog channel (top 5)
``/missrecall <q>`` semantic recall over misconception channel
``/save <id> <title>``  save a project (D channel)
``/end <pending>``  end the current session with a pending task (B)
``/clear <sid>``    nuke a student's memory (debug)
``/quit``           bye
==================  =================================================

Plain text (no leading ``/``) is treated as a student utterance and
flows through the full graph: recall → respond → extract → save.

Run
---
::

    uv run --env-file .env python chat_cli.py [--student alice]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Force MiMo extractor on by default for the CLI.
os.environ.setdefault("MEMORY_MODULE_USE_MIMO", "1")

from memory_module import schemas as S  # noqa: E402
from memory_module.service import MemoryService, build_default_service  # noqa: E402

# Reuse the LangGraph wiring from the demo so the CLI walks exactly the
# same recall → respond → extract → save path.
import langgraph_demo  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------- pretty printing helpers ----------


GREY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _print_help() -> None:
    print(_c(__doc__ or "", DIM))


def _print_state(state: S.StudentState) -> None:
    print(_c(f"\n— state({state.student_id}) —", BOLD))
    sess = state.session
    if sess.last_session_at:
        print(
            f"  B 会话续点: project={sess.current_project_id} "
            f"topic={sess.current_topic} pending={sess.pending_task} "
            f"(last={sess.last_session_at:%Y-%m-%d %H:%M})"
        )
    else:
        print("  B 会话续点: (空)")
    if state.mastery:
        print("  A 掌握度:")
        for m in state.mastery:
            print(f"    - {m.concept_id}: {m.score:.2f} (evidence={m.evidence_count})")
    else:
        print("  A 掌握度: (空)")
    if state.preferences:
        print("  E 偏好:")
        for p in state.preferences:
            print(f"    - {p.key} = {p.value} ({p.source})")
    else:
        print("  E 偏好: (空)")
    if state.projects:
        print("  D 项目:")
        for p in state.projects:
            print(f"    - {p.project_id}: {p.title}  highlights={p.highlights}  issues={p.issues}")
    else:
        print("  D 项目: (空)")
    if state.misconceptions:
        print("  F 迷思:")
        for mi in state.misconceptions:
            print(f"    - {mi.pattern_id} ×{mi.occurrences}: {mi.description}")
    else:
        print("  F 迷思: (空)")
    print()


def _print_recall(hits: list[S.RecallHit], label: str) -> None:
    if not hits:
        print(_c(f"  ({label} 无召回)", DIM))
        return
    print(_c(f"\n— recall.{label} —", BOLD))
    for h in hits:
        payload_snippet = (
            h.payload.get("summary")
            or h.payload.get("description")
            or str(h.payload)
        )
        if len(payload_snippet) > 100:
            payload_snippet = payload_snippet[:100] + "…"
        print(f"  score={h.score:+.3f}  {payload_snippet}")
    print()


# ---------- command handlers ----------


def cmd_state(svc: MemoryService, sid: str, _: list[str]) -> None:
    _print_state(svc.get_state(sid))


def cmd_who(svc: MemoryService, _sid: str, _: list[str]) -> None:
    # SQLiteStore doesn't expose list_students; cheap hack: peek the db.
    import sqlite3

    db_path = svc._store._db_path  # type: ignore[attr-defined]
    if db_path == ":memory:":
        print(_c("  (in-memory db; only current students live in this process)", DIM))
        return
    conn = sqlite3.connect(db_path)
    try:
        students: set[str] = set()
        for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ):
            try:
                rows = conn.execute(f"SELECT DISTINCT student_id FROM {table}").fetchall()
                students.update(r[0] for r in rows if r[0])
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()
    if students:
        print(_c("  students with memory: " + ", ".join(sorted(students)), DIM))
    else:
        print(_c("  (no students yet)", DIM))


def cmd_recall(svc: MemoryService, sid: str, args: list[str]) -> None:
    if not args:
        print(_c("  usage: /recall <query>", DIM))
        return
    q = " ".join(args)
    hits = svc.recall(sid, query=q, kind="dialog", top_k=5)
    _print_recall(hits, "dialog")


def cmd_missrecall(svc: MemoryService, sid: str, args: list[str]) -> None:
    if not args:
        print(_c("  usage: /missrecall <query>", DIM))
        return
    q = " ".join(args)
    hits = svc.recall(sid, query=q, kind="misconception", top_k=5)
    _print_recall(hits, "misconception")


def cmd_save(svc: MemoryService, sid: str, args: list[str]) -> None:
    if len(args) < 2:
        print(_c("  usage: /save <project_id> <title...>", DIM))
        return
    pid, title = args[0], " ".join(args[1:])
    svc.handle_event(
        sid,
        S.ProjectSavedEvent(project_id=pid, title=title),
    )
    print(_c(f"  saved project: {pid} = {title}", GREEN))


def cmd_end(svc: MemoryService, sid: str, args: list[str]) -> None:
    pending = " ".join(args) if args else None
    svc.handle_event(
        sid,
        S.SessionEndEvent(pending_task=pending),
    )
    print(_c(f"  session ended; pending={pending!r}", GREEN))


def cmd_clear(svc: MemoryService, _sid: str, args: list[str]) -> None:
    if not args:
        print(_c("  usage: /clear <student_id>   (irreversible)", DIM))
        return
    target = args[0]
    import sqlite3

    db_path = svc._store._db_path  # type: ignore[attr-defined]
    if db_path == ":memory:":
        print(_c("  in-memory db: restart the process to clear", DIM))
        return
    conn = sqlite3.connect(db_path)
    try:
        deleted = 0
        for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ):
            try:
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE student_id = ?", (target,)
                )
                deleted += cur.rowcount or 0
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()
    print(_c(f"  cleared {deleted} row(s) for {target}", YELLOW))


COMMANDS = {
    "/help": lambda *_: _print_help(),
    "/who": cmd_who,
    "/state": cmd_state,
    "/recall": cmd_recall,
    "/missrecall": cmd_missrecall,
    "/save": cmd_save,
    "/end": cmd_end,
    "/clear": cmd_clear,
}


# ---------- main loop ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="alice", help="active student id")
    parser.add_argument(
        "--db",
        default=os.environ.get("MEMORY_MODULE_DB", "./memory.db"),
        help="SQLite path (default ./memory.db; pass :memory: for ephemeral)",
    )
    args = parser.parse_args()

    if not os.environ.get("MIMO_API_KEY"):
        raise SystemExit("MIMO_API_KEY missing — run via `uv run --env-file .env ...`")

    # Build the service against the chosen db, then patch the demo
    # module's singleton so the LangGraph nodes see the same store.
    svc = build_default_service(db_path=args.db, use_mimo=True)
    langgraph_demo.SERVICE = svc

    graph = langgraph_demo.build_graph()
    sid = args.student

    print(_c(f"MiMo chat REPL ready. model={langgraph_demo.MIMO.model} db={args.db}", BOLD))
    print(_c(f"active student: {sid}    (try /help)", DIM))

    while True:
        try:
            line = input(_c(f"\n[{sid}] > ", CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in {"/quit", "/exit", ":q"}:
            break

        if line.startswith("/switch"):
            parts = line.split()
            if len(parts) < 2:
                print(_c("  usage: /switch <student_id>", DIM))
                continue
            sid = parts[1]
            print(_c(f"  switched to {sid}", GREEN))
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd, rest = parts[0], parts[1:]
            handler = COMMANDS.get(cmd)
            if not handler:
                print(_c(f"  unknown command: {cmd}  (try /help)", YELLOW))
                continue
            try:
                handler(svc, sid, rest)
            except Exception as exc:  # noqa: BLE001
                print(_c(f"  command failed: {exc}", YELLOW))
            continue

        # Plain text → run the graph.
        try:
            out = graph.invoke(
                {
                    "student_id": sid,
                    "student_utterance": line,
                    "extra_events": [],
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(_c(f"  graph error: {exc}", YELLOW))
            continue

        # Show what got recalled (compact).
        sess = (out.get("recalled_state") or {}).get("session") or {}
        if sess.get("pending_task") or sess.get("current_project_id"):
            print(
                _c(
                    f"  ↳ recall: project={sess.get('current_project_id')} "
                    f"pending={sess.get('pending_task')}",
                    DIM,
                )
            )
        recalled_dialog = out.get("recalled_dialog") or []
        if recalled_dialog:
            top = recalled_dialog[0]
            snippet = top["payload"].get("summary", "")[:80]
            print(_c(f"  ↳ top dialog hit (score={top['score']:.2f}): {snippet}", DIM))

        # Agent reply.
        reply = out.get("agent_reply") or ""
        print(_c(f"agent> {reply}", MAGENTA))

        # Show what got extracted + written.
        extracted = out.get("extracted_events") or []
        if extracted:
            for ev in extracted:
                print(_c(f"  ↪ wrote event: {ev}", GREY))

    print(_c("bye.", DIM))


if __name__ == "__main__":
    main()
