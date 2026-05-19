# Scratch 教学 Agent — 记忆模块（Demo）

为 Scratch 积木编程教学 agent 提供长期记忆服务的 FastAPI 应用。

> **要接 agent 看这个：[AGENT_INTEGRATION.md](AGENT_INTEGRATION.md)**
> 详细 HTTP 端点契约见 [INTEGRATION.md](INTEGRATION.md)。
> 本 README 讲设计与边界。

## 范围

**做**

- 6 类记忆的写入与查询：A 知识掌握 / B 会话续点 / C 对话历史 / D 项目档案 / E 偏好 / F 错误模式
- 多学生隔离（按 `student_id` 分片）
- HTTP API + 一个跑得通的 demo 脚本（自带断言）

**不做（v1 不在范围内）**

- LLM 上下文管理
- 认证、限流、家长后台、班级聚合
- 真正的教学 agent（demo 用模拟脚本喂事件）

## 架构

```
FastAPI Routes  →  MemoryService  →  MemoryStore   (Protocol)  →  SQLiteStore
                                  →  Embedder      (Protocol)  →  HashEmbedder / STEmbedder (BGE)
                                  →  Summarizer    (Protocol)  →  Stub / MiMoDialogSummarizer
                                  →  Detector      (Protocol)  →  Stub / MiMoMisconceptionDetector
```

存储、抽取、向量化都是 Protocol。**Stub 实现和真实现都已就位**，按需切换：

| 接口 | Stub 实现（默认） | 真实现（仓库里已有） | 升级路径（未实现） |
|---|---|---|---|
| `MemoryStore` | `SQLiteStore` | — | `PostgresStore` (pgvector) |
| `Embedder` | `HashEmbedder` | `STEmbedder`（BGE，`BAAI/bge-small-zh-v1.5`） | OpenAI / Cohere API embedder |
| `DialogSummarizer` | `StubDialogSummarizer` | `MiMoDialogSummarizer` | Claude / OpenAI summarizer |
| `MisconceptionDetector` | `StubMisconceptionDetector` | `MiMoMisconceptionDetector` | 其他 LLM |

切换方法：

```bash
uv sync --extra st              # 装 sentence-transformers，embedder 自动切到真 BGE
export MEMORY_MODULE_USE_MIMO=1 # 启用 MiMo 摘要 + 迷思聚类（同时需要 MIMO_API_KEY）
```

详细环境变量表与自定义 Protocol 实现示例见 [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) 第 8 节。

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

## API 表面

- `POST /students/{sid}/events` — 提交结构化事件（`mastery_update` / `session_end` / `project_saved` / `preference_set` / `error_observed`）
- `POST /students/{sid}/dialog` — 提交对话流，触发摘要写入
- `GET  /students/{sid}/state` — 学生完整画像
- `GET  /students/{sid}/recall?query=...&kind=dialog|misconception&top_k=5` — 语义检索

## 跑起来

```bash
uv sync                                          # 装依赖（首次）
uv run python demo.py                            # 端到端模拟场景（自带断言，无 LLM）
uv run python langgraph_demo.py                  # 加上真 LLM 的完整闭环（需要 .env）
uv run python chat_cli.py --student alice        # 交互式 REPL（持久化 ./memory.db）
uv run uvicorn memory_module.api:app --reload    # 起 HTTP 服务（可选）
```

## 后续演进

- `SQLiteStore` → `PostgresStore` (pgvector)
- 接入真实教学 agent
- 加认证、限流、异步抽取队列、家长/教师后台
