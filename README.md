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
FastAPI Routes  →  MemoryService  →  MemoryStore  (Protocol)  →  SQLiteStore
                                  →  Extractor    (Protocol)  →  StubExtractor
                                  →  Embedder     (Protocol)  →  HashEmbedder
```

存储、抽取、向量化都是 Protocol，默认走零依赖的最小实现，可平替为生产级实现：

| 接口 | demo 实现 | 升级路径 |
|---|---|---|
| `MemoryStore` | `SQLiteStore` | `PostgresStore` (pgvector) |
| `Embedder` | `HashEmbedder`（哈希向量） | `SentenceTransformerEmbedder` / API |
| `DialogSummarizer` | `StubDialogSummarizer`（拼接学生发言） | `LLMDialogSummarizer` |
| `MisconceptionDetector` | `StubMisconceptionDetector`（按 concept 计数） | `LLMMisconceptionDetector` |

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
- `HashEmbedder` → 真正的 embedding 模型
- `Stub*` → 接 LLM（Anthropic/OpenAI）做对话摘要与迷思聚类
- 接入真实教学 agent
- 加认证、限流、异步抽取队列、家长/教师后台
