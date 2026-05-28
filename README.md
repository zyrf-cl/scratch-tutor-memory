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
- 多学生隔离（按 `student_id` 分片）
- HTTP API + 一个跑得通的 demo 脚本（自带断言）

**不做（v1 不在范围内）**

- LLM 上下文管理
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

库模式（`from memory_module.service import build_default_service`）暴露同名方法，行为与 HTTP 一致。

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
uv run python scripts/smoke_milvus_store.py          # Milvus 后端（需先 uv sync --extra milvus 并起 Milvus）
uv run python scripts/smoke_mimo.py                  # MiMo LLM 连通性
```

## 环境变量

无需任何变量也能跑（默认 SQLite + Hash embedder + Stub extractors，纯本地无外部依赖）。完整开关如下：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_MODULE_DB` | `:memory:`（API 启动）/ `./memory.db`（demo） | SQLite 文件路径；`:memory:` 走纯内存 |
| `MEMORY_MODULE_STORE` | `sqlite` | `sqlite` \| `milvus`（后者需 `uv sync --extra milvus`） |
| `MEMORY_MODULE_EMBEDDER` | `bge` | `bge`/`st` \| `hash`/`stub` |
| `MEMORY_MODULE_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 任意 sentence-transformers 模型名 |
| `MEMORY_MODULE_EMBEDDING_DEVICE` | 自动 | `cuda` / `cpu` / `mps` |
| `MEMORY_MODULE_EMBEDDING_STRICT` | `0` | 设 `1` 时缺 `sentence-transformers` 直接 `RuntimeError`，不静默回退到 Hash |
| `MEMORY_MODULE_USE_MIMO` | `0` | 设 `1` 同时启用 MiMo 摘要 + 迷思聚类 + 情感判断（必须搭配 `MIMO_API_KEY`，否则**静默回退 Stub**） |
| `MEMORY_MODULE_MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址 |
| `MEMORY_MODULE_MILVUS_TOKEN` | — | Milvus 鉴权 `user:password` 或 token |
| `MEMORY_MODULE_MILVUS_DB` | — | Milvus database 名 |
| `MEMORY_MODULE_MILVUS_PREFIX` | `memory_module` | collection 前缀，多环境隔离 |
| `MEMORY_MODULE_MILVUS_TIMEOUT` | `5` | 操作超时秒 |
| `MEMORY_MODULE_MILVUS_CONSISTENCY` | `Strong` | 仅创建 collection 时生效 |
| `MIMO_API_KEY` | — | `USE_MIMO=1` 时必填 |
| `MIMO_BASE_URL` | `https://token-plan-cn.xiaomimimo.com/v1` | MiMo OpenAI 兼容端点 |
| `MIMO_MODEL` | `mimo-v2.5-pro` | MiMo 模型名 |

## Milvus 存储后端

SQLite 是默认后端，对话摘要 / 迷思 pattern 规模大时可切到 Milvus：

```bash
uv sync --extra milvus
export MEMORY_MODULE_STORE=milvus
export MEMORY_MODULE_MILVUS_URI=http://localhost:19530
uv run python scripts/smoke_milvus_store.py
```

注意：切换 embedder（如 Hash → BGE）会改变向量维度，Milvus collection 维度在创建时锁定。切换前需换 `MEMORY_MODULE_MILVUS_PREFIX` 或先删旧 collection；SQLite 后端则会**静默跳过**旧维度的向量。向量数据库提升的是检索规模和并发；召回质量仍取决于 embedder 与摘要质量。
