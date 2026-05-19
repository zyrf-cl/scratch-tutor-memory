"""Demo-able test for the PracticeAgent ↔ memory module integration.

Narrative
---------
两个学生（小红、小明）做同一道 Scratch loop 题目，各自跨 3 个 session：

  Session 1  小红首次做错 → agent 给第 1 级提示
  Session 2  小红同类错误再犯 → agent 看到记忆里"上次也是 WRONG_STRUCTURE/loop"
             自动升到第 2 级提示；同时记录她有点沮丧
  Session 3  打开新 session，先 load_snapshot
             → 屏幕上能看到：弱概念=loop / 上次 feedback_level=2 / 沮丧度≈0.8
  小明全程独立，证明跨学生隔离

每一步都打 print，能直接当 demo 给人看。

LLM
---
如果 .env 里有 MIMO_API_KEY/BASE_URL/MODEL，"agent 写给学生的那句话"会真调
MiMo 生成，并把记忆 snapshot 喂进 prompt —— 演示时能看到「同样错误、记忆
不同 → LLM 输出不同」。没 key 也能跑，退化为硬编码文案。

跑：
    uv run python demo_agent.py
    uv run --env-file .env python demo_agent.py   # 显式加载，更稳
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
AGENT_REPO = ROOT / "ai-programming-system-backend_hxh"
if str(AGENT_REPO) not in sys.path:
    sys.path.insert(0, str(AGENT_REPO))

os.environ.setdefault("MEMORY_MODULE_EMBEDDER", "hash")

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from agent.memory_client import AgentMemoryClient  # noqa: E402
from agent.practice_agent import classify_problem, update_memory  # noqa: E402
from agent.report import (  # noqa: E402
    AgentState,
    DiffReportData,
    EvaluationReport,
    EvaluationResultData,
)
from memory_module import schemas as S  # noqa: E402
from memory_module.mimo_client import MiMoClient  # noqa: E402


# ── LLM (optional) ───────────────────────────────────────────────────

_MIMO: MiMoClient | None = None

def _llm() -> MiMoClient | None:
    global _MIMO
    if _MIMO is not None:
        return _MIMO
    if not os.environ.get("MIMO_API_KEY"):
        return None
    try:
        _MIMO = MiMoClient()
        return _MIMO
    except Exception as e:
        print(f"  [warn] MiMo 初始化失败，本次走脚本文案：{e}")
        return None


def llm_compose_feedback(
    *,
    student_name: str,
    snapshot: S.AgentMemorySnapshot,
    error_types: list[str],
    diff_summary: str,
    score: float,
    feedback_level: int,
) -> tuple[str, float] | None:
    """让 MiMo 写给学生的那句话。返回 (内容, 耗时秒)。失败返回 None。"""
    client = _llm()
    if client is None:
        return None

    has_history = bool(snapshot.recent_error_types)
    affect = snapshot.affective_state or {}
    af_line = (
        f"frustration={affect.get('frustration', 0):.2f} / "
        f"confusion={affect.get('confusion', 0):.2f}"
        if affect else "（无记录）"
    )

    sys_msg = (
        "你是 Scratch 积木编程教学 agent。给小学生写反馈。"
        "要求：1) 不超过 60 个汉字；2) 一句话，不要罗列；"
        "3) feedback_level=1 只点拨方向，2 指出具体位置，3 给出明确动作；"
        "4) 如果学生 frustration 高，先安抚再讲技术；"
        "5) 称呼用学生名字。"
    )
    user_msg = (
        f"学生：{student_name}\n"
        f"当前评测：score={score:.2f}, 错误类型={error_types}, diff={diff_summary}\n"
        f"本轮 feedback_level={feedback_level}\n"
        f"记忆里的历史：\n"
        f"  recent_error_types={snapshot.recent_error_types}\n"
        f"  repeated_error_count={snapshot.repeated_error_count}\n"
        f"  weak_concepts={snapshot.weak_concepts}\n"
        f"  上一轮 feedback_level={snapshot.recent_feedback_level}\n"
        f"  情绪={af_line}\n"
        f"  有历史={has_history}\n"
        f"现在请写这句反馈。"
    )

    t0 = time.perf_counter()
    try:
        content = client.chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=2048,
        )
    except Exception as e:
        print(f"  [warn] MiMo 调用失败，回退脚本文案：{e}")
        return None
    return content.strip(), time.perf_counter() - t0





# ── pretty print helpers ─────────────────────────────────────────────

def banner(title: str) -> None:
    line = "═" * 64
    print(f"\n{line}\n  {title}\n{line}")

def step(title: str) -> None:
    print(f"\n── {title} " + "─" * (60 - len(title)))

def show_snapshot(label: str, snap: S.AgentMemorySnapshot) -> None:
    print(f"  [{label}]")
    print(f"    recent_feedback_level : {snap.recent_feedback_level}")
    print(f"    repeated_error_count  : {snap.repeated_error_count}")
    print(f"    recent_error_types    : {snap.recent_error_types}")
    print(f"    weak_concepts         : {snap.weak_concepts}")
    print(f"    mastered_concepts     : {snap.mastered_concepts}")
    af = snap.affective_state or {}
    if af:
        keys = ("frustration", "confidence", "engagement", "confusion")
        pairs = [f"{k}={af.get(k, 0):.2f}" for k in keys if k in af]
        print(f"    affective_state       : {', '.join(pairs)}")
    else:
        print(f"    affective_state       : (empty)")


# ── one attempt = one full PracticeAgent turn ───────────────────────

def run_attempt(
    client: AgentMemoryClient,
    *,
    user_id: str,
    student_name: str,
    session_id: str,
    task_id: str,
    passed: bool,
    score: float,
    error_types: list[str],
    diff_summary: str,
    fallback_feedback: str,
) -> AgentState:
    """跑完整一轮：load_snapshot → classify_problem → LLM 写反馈 → persist。

    student_feedback 优先用 MiMo 生成；MiMo 不可用时落回 fallback_feedback。
    """
    state = AgentState(
        task_id=task_id,
        user_id=user_id,
        session_id=session_id,
        project_name="loop-cat",
    )
    state.memory_snapshot = client.load_snapshot(
        user_id=user_id, session_id=session_id, task_id=task_id,
    )

    state.evaluation_result = EvaluationResultData(
        status="pass" if passed else "fail",
    )
    state.evaluation_report = EvaluationReport(
        project_name="loop-cat",
        passed=passed,
        score=score,
    )
    state.diff_report = DiffReportData(
        error_types=error_types,
        diff_summary=diff_summary,
        node_ids=["loop-block"],
        equivalence_score=score,
    )

    state = classify_problem(state)

    llm_out = llm_compose_feedback(
        student_name=student_name,
        snapshot=state.memory_snapshot,
        error_types=error_types,
        diff_summary=diff_summary,
        score=score,
        feedback_level=state.feedback_level,
    )
    if llm_out is not None:
        feedback, dt = llm_out
        print(f"  [LLM ✓ {dt:.1f}s] {feedback}")
        state.decision_result.student_feedback = feedback
    else:
        print(f"  [脚本文案] {fallback_feedback}")
        state.decision_result.student_feedback = fallback_feedback

    client.persist_state(update_memory(state))
    return state


# ── narrative ───────────────────────────────────────────────────────

def demo() -> None:
    banner("PracticeAgent × Memory Module — 可演示场景")
    llm_status = "已启用 (MiMo)" if _llm() is not None else "未启用 (无 MIMO_API_KEY)"
    print(f"  存储：纯内存 SQLite     LLM：{llm_status}")
    print("  学生：小红 (xiaohong) / 小明 (xiaoming)")
    print("  任务：loop-cat —— 用 forever 让小猫一直跑")

    client = AgentMemoryClient(db_path=":memory:", project_root=AGENT_REPO)

    # ── Session 1 ───────────────────────────────────────────────
    banner("Session 1：两个学生首次尝试，记忆都是空的")

    step("小红 load_snapshot（应该全空）")
    snap = client.load_snapshot(
        user_id="xiaohong", session_id="s1", task_id="loop-cat",
    )
    show_snapshot("before classify", snap)

    step("小红 提交 —— 漏写循环，WRONG_STRUCTURE")
    s1_h = run_attempt(
        client,
        user_id="xiaohong", student_name="小红",
        session_id="s1", task_id="loop-cat",
        passed=False, score=0.4,
        error_types=["WRONG_STRUCTURE"],
        diff_summary="missing forever block around motion",
        fallback_feedback="小红，注意小猫只动了一次，要怎么让它一直动呢？",
    )
    print(f"  → classify_problem 决定的提示级别：feedback_level = {s1_h.feedback_level}")

    step("小明 提交 —— 同样的错（独立路径）")
    s1_m = run_attempt(
        client,
        user_id="xiaoming", student_name="小明",
        session_id="s1", task_id="loop-cat",
        passed=False, score=0.4,
        error_types=["WRONG_STRUCTURE"],
        diff_summary="missing forever block",
        fallback_feedback="小明，再看看循环积木～",
    )
    print(f"  → 小明 feedback_level = {s1_m.feedback_level}   （也是 1，独立计数）")

    # ── Session 2 ───────────────────────────────────────────────
    banner("Session 2：小红又来了，再次同类错误 → 期望升级")

    step("小红 进入新 session，load_snapshot")
    snap = client.load_snapshot(
        user_id="xiaohong", session_id="s2", task_id="loop-cat",
    )
    show_snapshot("agent 记忆里的小红", snap)
    print("  ↑ 注意：上一轮的 WRONG_STRUCTURE 已经在记忆里了；这条 snapshot 会喂给 LLM")

    step("小红 再次提交 —— 还是 WRONG_STRUCTURE（看 LLM 这次怎么说）")
    s2_h = run_attempt(
        client,
        user_id="xiaohong", student_name="小红",
        session_id="s2", task_id="loop-cat",
        passed=False, score=0.45,
        error_types=["WRONG_STRUCTURE"],
        diff_summary="still missing the forever wrapper",
        fallback_feedback="(更具体的第 2 级提示，指出 forever 积木的位置)",
    )
    print(f"  → feedback_level 升到：{s2_h.feedback_level}   ← 由记忆驱动，不是凭空")
    assert s2_h.feedback_level >= 2, "期望升到 ≥2 级"

    step("（自动）persist_state 已经按「失败 + 同类重复错误 + 升级到 2 级」")
    print("       推断了一条情感事件 —— 不用调用方手写，掉到 memory 里了。")

    # ── Session 3 ───────────────────────────────────────────────
    banner("Session 3：第二天小红回来，看 agent 还记得什么")

    step("小红 load_snapshot —— 这是 agent 写下一句话前看到的全部记忆")
    snap_h = client.load_snapshot(
        user_id="xiaohong", session_id="s3", task_id="loop-cat",
    )
    show_snapshot("小红", snap_h)

    print("\n  ✓ recent_feedback_level=2  → 这次开口直接从 2 级起，不要又从 1 级讲")
    print("  ✓ weak_concepts 含 'loop'  → 教学策略要把 loop 当弱点")
    print("  ✓ frustration 明显抬高   → 措辞要鼓励，不要再压力")
    print("  ✓ 同一个 task 重复 error  → 该考虑换种讲法 / 给例子")

    if _llm() is not None:
        step("对照实验：LLM 在 session 1 (无历史) vs session 2 (有历史) 写的两句话")
        print(f"  session 1（无历史，level=1）：")
        print(f"    {s1_h.decision_result.student_feedback}")
        print(f"  session 2（有同类错误历史，level=2，记忆触发了沮丧）：")
        print(f"    {s2_h.decision_result.student_feedback}")
        print("  ↑ 同样的错误类型，仅仅因为记忆变了，LLM 的反馈就变了 ——")
        print("    这就是「记忆模块接进 agent」对 LLM 输出的真实影响。")

    step("对比：小明的快照（应当完全独立）")
    snap_m = client.load_snapshot(
        user_id="xiaoming", session_id="s3", task_id="loop-cat",
    )
    show_snapshot("小明", snap_m)
    print("  ✓ 小明 recent_error_types 只有 1 条（session 1 那次），没沾上小红的第 2 次")
    print("  ✓ 小明 repeated_error_count=0，仍是初学者画像")

    # ── 断言（演示不能只看脸，得真过 check） ────────────────
    banner("自检断言")
    checks = [
        ("小红 feedback_level 在 s2 升级", s2_h.feedback_level >= 2),
        ("小红 weak_concepts 含 loop", "loop" in snap_h.weak_concepts),
        ("小红 recent_error_types 全是 WRONG_STRUCTURE",
         set(snap_h.recent_error_types) == {"WRONG_STRUCTURE"}),
        ("小红 frustration 明显高于 baseline (>0.4)",
         snap_h.affective_state.get("frustration", 0) > 0.4),
        ("小明与小红隔离 —— 小明只有 1 条错误记录，未重复",
         len(snap_m.recent_error_types) == 1 and snap_m.repeated_error_count == 0
         and len(snap_h.recent_error_types) == 2 and snap_h.repeated_error_count >= 1),
        ("小明在 s1 仍是 level 1", s1_m.feedback_level == 1),
    ]
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")
    assert all(ok for _, ok in checks), "有断言失败"

    print("\n演示结束。所有断言通过。\n")


if __name__ == "__main__":
    demo()
