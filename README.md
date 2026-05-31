# Scratch 教学 Agent — 记忆模块（Demo）

为 Scratch 积木编程教学 agent 提供长期记忆服务的 FastAPI 应用。

> **要接 agent 看这个：[AGENT_INTEGRATION.md](AGENT_INTEGRATION.md)**
> 详细 HTTP 端点契约见 [INTEGRATION.md](INTEGRATION.md)。
> 本 README 讲设计与边界。

## 范围

**做**

- 6 类记忆的写入与查询：A 知识掌握 / B 会话续点 / C 对话历史 / D 项目档案 / E 偏好 / F 错误模式
- 面向教学 agent 的诊断记忆：`recent_feedback_level` / `weak_concepts` / `affective_state`
- 服务端情感信号写入与聚合：支持规则路径，启用 MiMo 后可自动做 LLM 情感判断
- 短期上下文管理（**独立解耦功能**）：接收 agent 的运行对话，按 token / 轮次预算取回窗口；分 session 级与用户级（跨 session）两条独立流；溢出部分默认截断，启用 Kimi 后可改为 LLM 摘要压缩（见「上下文管理」一节）
- 多学生隔离（按 `student_id` 分片）
- HTTP API + 一个跑得通的 demo 脚本（自带断言）

**不做（v1 不在范围内）**

- LLM 上下文的自动摘要压缩**作为默认行为**（短期上下文默认只做新近度窗口化截断；基于 LLM 的溢出摘要已实现但需显式开启 `MEMORY_MODULE_USE_KIMI=1`，见「上下文管理」一节）
- 认证、限流、家长后台、班级聚合
- 真正的教学 agent（demo 用模拟脚本喂事件）

## 架构

```
FastAPI Routes  →  MemoryService  →  MemoryStore   (Protocol)  →  SQLiteStore / MilvusStore
                                  →  Embedder      (Protocol)  →  HashEmbedder / STEmbedder (BGE)
                                  →  Summarizer    (Protocol)  →  Stub / MiMoDialogSummarizer
                                  →  Detector      (Protocol)  →  Stub / MiMoMisconceptionDetector
                                  →  Affective     (Protocol)  →  Stub / MiMoAffectiveSignalDetector
```

存储、抽取、向量化都是 Protocol，Stub 实现和真实现都已在仓库中，按需切换：

| 接口 | Stub 实现（默认） | 真实现（已就位） |
|---|---|---|
| `MemoryStore` | `SQLiteStore`（`memory_module/store.py`） | `MilvusStore`（`memory_module/milvus_store.py`） |
| `Embedder` | `HashEmbedder` | `STEmbedder`（`BAAI/bge-small-zh-v1.5`，依赖 `extra st`） |
| `DialogSummarizer` | `StubDialogSummarizer` | `MiMoDialogSummarizer`（需 `MIMO_API_KEY`） |
| `MisconceptionDetector` | `StubMisconceptionDetector` | `MiMoMisconceptionDetector`（需 `MIMO_API_KEY`） |
| `AffectiveSignalDetector` | `StubAffectiveSignalDetector` | `MiMoAffectiveSignalDetector`（需 `MIMO_API_KEY`） |

切换方法见下方「环境变量」一节，详细自定义 Protocol 实现示例见 [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) 第 8 节。

## 6 类记忆

| 类型 | 字段 | 写入路径 | 主要查询 |
|---|---|---|---|
| A 掌握度 | `concept_id, score(0~1), evidence_count` | 显式 `mastery_update` 事件 | 列表 / 阈值过滤 |
| B 会话续点 | `current_project, topic, pending_task` | 显式 `session_end` 事件 | 续课读取 |
| C 对话历史 | `summary, embedding` | dialog → summarizer | 语义检索 |
| D 项目档案 | `title, structure, highlights, issues` | 显式 `project_saved` | 列表 |
| E 偏好 | `key, value, source` | 显式 `preference_set` | 列表 |
| F 迷思 | `pattern_id, description, occurrences` | `error_observed` → 聚类 | 列表 / 语义检索 |

Concept 是预定义闭集：`loop / conditional / event / variable / broadcast / clone / sprite / sound / motion / sensing / operator / list`。

此外，面向教学 agent 还维护一组诊断态快照：`recent_feedback_level`、`weak_concepts`、`affective_state`。其中 `affective_state` 聚合最近情感信号（如 frustration / confusion / confidence / engagement），用于跨 session 调整提示强度与反馈语气。

## API 表面

- `GET  /health` — 健康检查，返回 `{"status": "ok"}`
- `POST /students/{sid}/events` — 提交结构化事件（`mastery_update` / `session_end` / `project_saved` / `preference_set` / `error_observed`）
- `POST /students/{sid}/dialog` — 提交对话流（`turns: list[{role, content}]`），服务端摘要 + 向量化入库
- `GET  /students/{sid}/state` — 学生完整画像（6 通道）
- `GET  /students/{sid}/recall?query=...&kind=dialog|misconception&top_k=5` — 向量语义检索（HTTP 强制 `1 ≤ top_k ≤ 50`）
- `POST /students/{sid}/agent-turns` — 写入一次 agent 诊断回合，服务端**自动派生情感事件**并在失败且有 `concept_id` 时**自动写一条 `error_observed`**
- `GET  /students/{sid}/agent-memory?session_id=...&task_id=...` — 读取 agent 决策快照（`recent_feedback_level` / `weak_concepts` / `affective_state` 等 7 字段）
- `POST /students/{sid}/affective-events` — 显式写入情感信号（外部已有情感判别器时使用；正常 agent-turn 路径无需调）
- **上下文管理（独立功能，与上面解耦）**：两条独立流——session 级 `…/context/{session_id}[/messages|/stats]` 与用户级 / 跨 session `…/user-context[/messages|/stats]`（只认 `student_id`）— 详见下方「上下文管理」一节

库模式（`from memory_module.service import build_default_service`）暴露同名方法，行为与 HTTP 一致。

## 上下文管理（独立功能）

短期「工作上下文」缓冲，和上面 6+1 类长期记忆**解耦**：agent 把正在进行的对话推进来，按预算取回一个窗口塞回给 LLM。它和 C 通道（对话历史 → 摘要 + 向量化的**长期**语义记忆）是两个不同的东西——这里存的是原始运行上下文，按新近度窗口化，不做摘要、不做向量检索。

它分两条**互不共享状态**的独立流，按 `student_id` 隔离：

- **session 级**（`/context/{session_id}`）：单条对话线程的工作上下文，存 `context_message` 表。
- **用户级 / 跨 session**（`/user-context`）：跨该用户所有 session 的上下文（长期目标、稳定偏好提示等），存独立的 `user_context_message` 表，接口只认 `student_id`、不带 session。

两条流复用同一套窗口化/压缩逻辑与消息模型，但各自独立的表、store、service、router——互不可见，清空一条不影响另一条。

- 自带独立子包 `memory_module/context/`（`schemas` / `store` / `compaction` / `service` / `api`），两张独立表 `context_message`（session 级）与 `user_context_message`（用户级），与 `MemoryService` 不共享任何状态；在 `build_app` 装配处挂两个 router（session / user-context）。
- 默认与主库同文件（`MEMORY_MODULE_DB`），可用 `MEMORY_MODULE_CONTEXT_DB` 拆到单独文件。
- 窗口化策略由可插拔的 `Compactor` 决定，默认 `TruncatingCompactor`：`pinned` 消息（如 system prompt）永不裁掉，其余按「新 → 旧」填到 `max_tokens` / `max_turns` 预算用满为止；不传预算则全量返回。开启 `MEMORY_MODULE_USE_KIMI=1`（+`KIMI_API_KEY`，Kimi/Moonshot OpenAI 兼容端点）后换成 `LLMSummarizingCompactor`：溢出的旧消息不丢弃，而是用 LLM 压成一条合成 `system` 摘要消息插回窗口开头，并在 `compaction_note` 里说明；摘要失败时自动退回截断。两者接口一致，service 与 HTTP API 不变。

8 个端点（两条独立流，与核心 7 端点解耦）：

**session 级**（`/context/{session_id}`）：

| 方法 / 路径 | 说明 | 返回 |
|---|---|---|
| `POST   /students/{sid}/context/{session_id}/messages` | 追加上下文消息（`role` / `content` / `pinned` / `token_count?` / `metadata?`） | `{appended, total_messages, total_tokens}` |
| `GET    /students/{sid}/context/{session_id}?max_tokens=&max_turns=` | 取窗口（保留 pinned + 最近消息） | `{messages, total_messages, returned_messages, total_tokens, truncated, compaction_note}` |
| `GET    /students/{sid}/context/{session_id}/stats` | 计数与 token 统计 | `{message_count, total_tokens, last_updated}` |
| `DELETE /students/{sid}/context/{session_id}` | 清空该会话上下文 | `{deleted}` |

**用户级 / 跨 session**（`/user-context`，接口只认 `student_id`、无 session 段，返回里 `session_id` 恒为 `null`）：

| 方法 / 路径 | 说明 | 返回 |
|---|---|---|
| `POST   /students/{sid}/user-context/messages` | 追加用户级上下文消息（字段同 session 级） | `{appended, total_messages, total_tokens}` |
| `GET    /students/{sid}/user-context?max_tokens=&max_turns=` | 取窗口（保留 pinned + 最近消息） | 同上窗口结构 |
| `GET    /students/{sid}/user-context/stats` | 计数与 token 统计 | `{message_count, total_tokens, last_updated}` |
| `DELETE /students/{sid}/user-context` | 清空该用户的跨 session 上下文 | `{deleted}` |

库模式：

```python
from memory_module.context import (
    build_default_context_service,
    build_default_user_context_service,
    ContextMessage,
)

# session 级：单条对话线程
ctx = build_default_context_service(db_path="./memory.db")
ctx.append("alice", "sess-1", [
    ContextMessage(role="system", content="你是 Scratch 老师", pinned=True),
    ContextMessage(role="user", content="怎么让小猫跳起来？"),
])
window = ctx.get_window("alice", "sess-1", max_tokens=2000)

# 用户级 / 跨 session：只认 student_id，无 session 段
uctx = build_default_user_context_service(db_path="./memory.db")
uctx.append("alice", [
    ContextMessage(role="system", content="长期目标：做一个跑酷游戏", pinned=True),
])
user_window = uctx.get_window("alice", max_tokens=2000)
```

## 跑起来

```bash
uv sync                                              # 装依赖（首次）
uv run python demo.py                                # 端到端模拟场景（含断言，全 Stub，无 LLM）
uv run python demo_agent.py                          # PracticeAgent ↔ 记忆模块 demo（2 学生 × 3 sessions，含跨会话记忆生效）
uv run python langgraph_demo.py                      # LangGraph 完整闭环（需要 .env 提供 MIMO_API_KEY）
uv run python chat_cli.py --student alice            # 交互式 REPL，持久化到 ./memory.db
uv run uvicorn memory_module.api:app --reload        # 起 HTTP 服务（库模式直接 import 即可，不一定需要起服务）
```

冒烟脚本（手动验证子系统是否正常）：

```bash
uv run python scripts/smoke_agent_memory.py          # agent-turn / affective-event / agent-memory 链路
uv run python scripts/smoke_context.py               # 上下文管理（append / window / stats / clear，独立功能）
uv run python scripts/smoke_milvus_store.py          # Milvus 后端（需先 uv sync --extra milvus 并起 Milvus）
uv run python scripts/smoke_mimo.py                  # MiMo LLM 连通性
```

## 环境变量

无需任何变量也能跑（默认 SQLite + Hash embedder + Stub extractors，纯本地无外部依赖）。完整开关如下：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_MODULE_DB` | `:memory:`（API 启动）/ `./memory.db`（demo） | SQLite 文件路径；`:memory:` 走纯内存 |
| `MEMORY_MODULE_CONTEXT_DB` | 跟随 `MEMORY_MODULE_DB` | 上下文管理（独立功能）的存储文件；设了就把 `context_message` / `user_context_message` 两张表拆到单独的库 |
| `MEMORY_MODULE_CONTEXT_MAX_TOKENS` | — | 取上下文窗口时的默认 token 预算（请求带 `max_tokens` 可覆盖） |
| `MEMORY_MODULE_CONTEXT_MAX_TURNS` | — | 取上下文窗口时的默认轮次预算（请求带 `max_turns` 可覆盖） |
| `MEMORY_MODULE_STORE` | `sqlite` | `sqlite` \| `milvus`（后者需 `uv sync --extra milvus`） |
| `MEMORY_MODULE_EMBEDDER` | `bge` | `bge`/`st` \| `hash`/`stub` |
| `MEMORY_MODULE_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 任意 sentence-transformers 模型名 |
| `MEMORY_MODULE_EMBEDDING_DEVICE` | 自动 | `cuda` / `cpu` / `mps` |
| `MEMORY_MODULE_EMBEDDING_STRICT` | `0` | 设 `1` 时缺 `sentence-transformers` 直接 `RuntimeError`，不静默回退到 Hash |
| `MEMORY_MODULE_USE_MIMO` | `0` | 设 `1` 同时启用 MiMo 摘要 + 迷思聚类 + 情感判断（必须搭配 `MIMO_API_KEY`，否则**静默回退 Stub**） |
| `MEMORY_MODULE_USE_KIMI` | `0` | 设 `1` 启用上下文溢出的 LLM 摘要压缩（`LLMSummarizingCompactor`，仅作用于「上下文管理」功能；必须搭配 `KIMI_API_KEY`/`MOONSHOT_API_KEY`，否则**静默回退截断**） |
| `MEMORY_MODULE_MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址 |
| `MEMORY_MODULE_MILVUS_TOKEN` | — | Milvus 鉴权 `user:password` 或 token |
| `MEMORY_MODULE_MILVUS_DB` | — | Milvus database 名 |
| `MEMORY_MODULE_MILVUS_PREFIX` | `memory_module` | collection 前缀，多环境隔离 |
| `MEMORY_MODULE_MILVUS_TIMEOUT` | `5` | 操作超时秒 |
| `MEMORY_MODULE_MILVUS_CONSISTENCY` | `Strong` | 仅创建 collection 时生效 |
| `MIMO_API_KEY` | — | `USE_MIMO=1` 时必填 |
| `MIMO_BASE_URL` | `https://token-plan-cn.xiaomimimo.com/v1` | MiMo OpenAI 兼容端点 |
| `MIMO_MODEL` | `mimo-v2.5-pro` | MiMo 模型名 |
| `KIMI_API_KEY` | — | `USE_KIMI=1` 时必填（亦可用 `MOONSHOT_API_KEY`） |
| `KIMI_BASE_URL` | `https://api.moonshot.cn/v1` | Kimi/Moonshot OpenAI 兼容端点（国际账号用 `https://api.moonshot.ai/v1`） |
| `KIMI_MODEL` | `moonshot-v1-8k` | Kimi 模型名 |

## Milvus 存储后端

SQLite 是默认后端，对话摘要 / 迷思 pattern 规模大时可切到 Milvus：

```bash
uv sync --extra milvus
export MEMORY_MODULE_STORE=milvus
export MEMORY_MODULE_MILVUS_URI=http://localhost:19530
uv run python scripts/smoke_milvus_store.py
```

注意：切换 embedder（如 Hash → BGE）会改变向量维度，Milvus collection 维度在创建时锁定。切换前需换 `MEMORY_MODULE_MILVUS_PREFIX` 或先删旧 collection；SQLite 后端则会**静默跳过**旧维度的向量。向量数据库提升的是检索规模和并发；召回质量仍取决于 embedder 与摘要质量。
