# Memory Module — Agent 对接指南

> 给 Scratch 教学 agent 开发那边看的。
> 这份文档只讲 **怎么用**，不讲实现。实现细节看 `README.md` 和源码。

---

## 一、它是什么

一个独立的 HTTP 服务，给你的教学 agent 提供「长期记忆」。

你只管两件事：

1. **写**：每次学生有动作（答对/答错/保存项目/说了偏好/会话结束），发一条事件给它。
2. **读**：每次开新会话或要决定下一句话怎么说之前，问它「这个学生你记得什么」。

它会替你管 6 类记忆、做语义检索、做学生隔离。默认持久化到 SQLite，也可以切到 Milvus。

```
你的 Agent  ──HTTP──▶  Memory Module  ──▶  SQLite (memory.db) / Milvus
        ◀──state/recall──
```

---

## 二、3 分钟跑起来

```bash
# 1. 装依赖
uv sync

# 2. 起服务（默认监听 127.0.0.1:8000，持久化到 ./memory.db）
uv run uvicorn memory_module.api:app --port 8000

# 3. 自检
curl http://127.0.0.1:8000/health
# -> {"status":"ok"}

# 4. 在线 API 文档（Swagger）
# 浏览器打开 http://127.0.0.1:8000/docs
```

**改存储位置**：

```bash
MEMORY_MODULE_DB=/var/data/memory.db uv run uvicorn memory_module.api:app
# 传 :memory: 走纯内存（重启就没了，只用于测试）
```

**切到 Milvus**：

```bash
uv sync --extra milvus
MEMORY_MODULE_STORE=milvus MEMORY_MODULE_MILVUS_URI=http://localhost:19530 \
  uv run uvicorn memory_module.api:app
```

Milvus 适合大量对话摘要/误区向量召回；小规模本地验证继续用 SQLite 更简单。

---

## 三、7 个 HTTP 端点

`student_id` 自己定，字符串就行（学号/uuid/微信 openid 都可以）。**不同 student_id 之间完全隔离。**

### 1. `POST /students/{student_id}/events` — 写结构化事件

请求体永远是 `{"event": {...}}` 这种信封形态。`type` 字段决定走哪个通道。

详见下一节「五种事件」。返回 `{"status": "ok"}`。

### 2. `POST /students/{student_id}/dialog` — 写对话流

```json
{
  "turns": [
    {"role": "agent",   "content": "今天我们让小猫动起来"},
    {"role": "student", "content": "我想让小猫一直跑"}
  ]
}
```

服务端会自动摘要 + 向量化 + 入库。返回 `{"turn_id": 7}`。

**调用时机**：每一轮 agent↔student 对话结束后，把这一轮的 turns 推过来。不要等积攒一大串，越长摘要质量越差。

### 3. `GET /students/{student_id}/state` — 拿学生完整画像

无参数。返回结构（关键字段）：

```json
{
  "student_id": "alice",
  "session": {
    "last_session_at": "2026-05-14T10:30:00Z",
    "current_project_id": "proj-1",
    "current_topic": "动画",
    "pending_task": "下次解决小猫飞出屏幕"
  },
  "mastery": [
    {"concept_id": "loop",  "score": 0.65, "evidence_count": 4, "last_updated": "..."}
  ],
  "preferences":   [{"key": "topic", "value": "猫和狗的动画", "source": "self", ...}],
  "projects":      [{"project_id": "proj-1", "title": "跑步小猫", ...}],
  "misconceptions":[{"pattern_id": "concept:event", "occurrences": 2, ...}]
}
```

**调用时机**：每个新会话开始时调用一次；每次构造 prompt 前可以再调一次拿最新值。

### 4. `GET /students/{student_id}/recall?query=...&kind=dialog&top_k=5` — 语义检索

- `kind=dialog`：在历史对话摘要里搜
- `kind=misconception`：在迷思模式里搜
- `top_k`：1–50

返回 `[{"kind": "...", "score": 0.83, "payload": {...}}, ...]`，按相关性降序。

**调用时机**：学生当前发言可能勾起历史话题时（比如"那个小猫的项目"），用学生发言当 query 拉相关历史。

### 5. `POST /students/{student_id}/agent-turns` — 写一次 agent 诊断回合

请求体示例：

```json
{
  "session_id": "s2",
  "task_id": "loop-cat",
  "attempt_id": "a-002",
  "project_name": "loop-cat",
  "passed": false,
  "score": 0.35,
  "primary_error_type": "WRONG_STRUCTURE",
  "concept_id": "loop",
  "feedback_level": 2,
  "feedback_text": "先找 forever 积木，再看运动积木是不是放进去了。",
  "user_text": "我还是不会，这个太难了",
  "diff_summary": "missing forever block around motion",
  "highlight_node_ids": ["block-1"],
  "need_rag": false,
  "need_repair_validation": false
}
```

服务端会自动：记录最近错误与反馈等级、推断薄弱概念、生成一条情感事件；启用 `MEMORY_MODULE_USE_MIMO=1` 时优先走 MiMo 做服务端情感判断。返回 `{"turn_id": 12}`。

### 6. `GET /students/{student_id}/agent-memory?session_id=...&task_id=...` — 读取 agent 决策快照

返回结构（关键字段）示例：

```json
{
  "recent_feedbacks": ["先看 forever 放的位置"],
  "recent_error_types": ["WRONG_STRUCTURE"],
  "recent_feedback_level": 2,
  "repeated_error_count": 0,
  "mastered_concepts": [],
  "weak_concepts": ["loop"],
  "affective_state": {
    "frustration": 0.53,
    "confidence": 0.22,
    "engagement": 0.5,
    "confusion": 0.63
  }
}
```

**调用时机**：生成下一句反馈前，或跨 session 恢复某个 task 的教学状态前。

### 7. `POST /students/{student_id}/affective-events` — 可选：显式写情感信号

当你已经有外部情感识别器、传感器或前端埋点时，可以单独写入情感信号。否则通常只写 `/agent-turns` 就够了，避免双写。

请求体示例：

```json
{
  "session_id": "s2",
  "task_id": "loop-cat",
  "frustration_score": 0.8,
  "confusion_score": 0.7,
  "confidence_score": 0.2,
  "engagement_score": 0.5,
  "source": "agent_inferred",
  "evidence": "连续两次同类错误"
}
```

返回 `{"event_id": 5}`。

---

## 三·补、上下文管理（独立功能）

> 这是和上面 7 个端点**解耦**的独立功能：短期「工作上下文」缓冲——你把正在进行的对话推进来，按预算取回一个窗口塞给 LLM。它和 C 通道（对话历史 → 摘要 + 向量化的**长期**记忆）是两回事，互不影响。

它分两条**互不共享状态**的独立流（各自独立的表 / store / service / router，`student_id` 都做隔离分片）：

- **session 级**（`/context/{session_id}`）：单条对话线程的工作上下文，`session_id` 标识线程，存 `context_message` 表。
- **用户级 / 跨 session**（`/user-context`）：跨该用户所有 session 的上下文（长期目标、稳定偏好提示等），接口只认 `student_id`、不带 session，存独立的 `user_context_message` 表，窗口/统计返回里 `session_id` 恒为 `null`。

### C1. `POST /students/{sid}/context/{session_id}/messages` — 追加上下文消息

```json
{
  "messages": [
    {"role": "system", "content": "你是 Scratch 编程老师", "pinned": true},
    {"role": "user", "content": "怎么让小猫跳起来？"},
    {"role": "assistant", "content": "用「change y by」积木……"}
  ]
}
```

- `role`：`system` / `user` / `assistant` / `tool`
- `pinned`：`true` 的消息（如 system prompt）取窗口时**永不被裁掉**
- `token_count`：可选；传了用你的精确值，不传服务端按 `len/4` 粗估
- `metadata`：可选 dict

返回 `{"appended": 3, "total_messages": 3, "total_tokens": 42}`。

### C2. `GET /students/{sid}/context/{session_id}?max_tokens=2000&max_turns=20` — 取窗口

- 不带预算 → 返回全部消息（按时间顺序）
- 带 `max_tokens` / `max_turns` → 保留全部 `pinned`，再用**最近**的消息按「新→旧」填，直到任一预算用满；结果按时间顺序返回

```json
{
  "student_id": "alice", "session_id": "sess-1",
  "messages": [
    {"id": 1, "seq": 1, "role": "system", "content": "...", "pinned": true,
     "token_count": 8, "created_at": "..."}
  ],
  "total_messages": 12, "returned_messages": 6, "total_tokens": 1980,
  "truncated": true, "compaction_note": null
}
```

`truncated=true` 表示有较旧消息被窗口裁掉。`compaction_note` 预留给未来的 LLM 摘要压缩器，当前恒为 `null`。

### C3. `GET /students/{sid}/context/{session_id}/stats` — 统计

返回 `{"student_id", "session_id", "message_count", "total_tokens", "last_updated"}`。

### C4. `DELETE /students/{sid}/context/{session_id}` — 清空该线程上下文

返回 `{"deleted": 12}`。

### C5–C8. 用户级 / 跨 session（`/user-context`，无 session 段）

和 C1–C4 字段、窗口语义完全一致，区别只有：路径不带 `session_id`，按 `student_id` 维护一条**跨该用户所有 session**的上下文流；窗口/统计返回里 `session_id` 恒为 `null`。适合放长期目标、稳定偏好提示这类跨课时都该带上的内容。

- `POST   /students/{sid}/user-context/messages` — 追加（body 同 C1）→ `{"appended", "total_messages", "total_tokens"}`
- `GET    /students/{sid}/user-context?max_tokens=&max_turns=` — 取窗口（结构同 C2，`session_id` 为 `null`）
- `GET    /students/{sid}/user-context/stats` — 统计 → `{"student_id", "session_id": null, "message_count", "total_tokens", "last_updated"}`
- `DELETE /students/{sid}/user-context` — 清空该用户的跨 session 上下文 → `{"deleted": N}`

**库模式**（同进程可跳过 HTTP）：

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

环境变量：`MEMORY_MODULE_CONTEXT_DB`（把 `context_message` / `user_context_message` 两张表拆到独立库文件）、`MEMORY_MODULE_CONTEXT_MAX_TOKENS` / `MEMORY_MODULE_CONTEXT_MAX_TURNS`（默认窗口预算，可被请求参数覆盖）。

---

## 四、五种事件 — 写入路径

每种事件都包成 `{"event": <下面这些>}` 发到 `POST /students/{sid}/events`。

### A. `mastery_update` — 掌握度（一个概念学会到几成）

```json
{"type": "mastery_update", "concept_id": "loop", "score": 0.7, "evidence_note": "独立用 forever 完成动画"}
```

- `score` 必须在 `[0.0, 1.0]`
- `concept_id` **必须**从白名单里选（见下文）
- `evidence_note` 可选，纯文本

**什么时候发**：你判断学生在这个概念上有新的证据时。不需要每轮都发。

### B. `session_end` — 会话结束

```json
{"type": "session_end", "current_project_id": "proj-1", "current_topic": "动画", "pending_task": "下次解决小猫飞出屏幕"}
```

三个字段都可选。但 `pending_task` 是续课关键，强烈建议填。

**什么时候发**：学生点退出/超时/老师手动结课。**每个会话结束发一次**。

### C. （对话）走 `POST /dialog`，不走 events。

### D. `project_saved` — 项目存盘

```json
{
  "type": "project_saved",
  "project_id": "proj-1",
  "title": "跑步小猫",
  "structure": {"sprites": 1, "scripts": 2},
  "highlights": ["首次用 forever"],
  "issues": ["小猫飞出屏幕"]
}
```

`project_id` / `title` 必填，其余可选。

**什么时候发**：学生在 Scratch 里点保存、你的 agent 帮他存档时。

### E. `preference_set` — 偏好

```json
{"type": "preference_set", "key": "topic", "value": "猫和狗的动画", "source": "self"}
```

- `source` 只能是 `"self"`（学生主动说的）或 `"inferred"`（你 agent 推断的）
- 同一个 `key` 后写覆盖前写

**什么时候发**：学生明确表达喜好（"我想做动画"），或你从行为里推断出稳定偏好。

### F. `error_observed` — 出错观察

```json
{"type": "error_observed", "concept_id": "event", "description": "把 when-flag-clicked 当成 forever 用了", "context": "可选上下文"}
```

`concept_id` 可选但强烈建议填——同一个 concept 多次出错会被自动聚类成「迷思 pattern」(F 通道)。

**什么时候发**：你确认学生在某个概念上犯错时。不要把"打错字"这种发上来。

---

## 五、概念白名单（重要）

`concept_id` 字段只能从这 12 个里选：

```
loop  conditional  event  variable  broadcast  clone
sprite  sound  motion  sensing  operator  list
```

发别的字符串会被服务端拒绝（400）。如果你需要新增，找模块维护者改 `memory_module/concepts.py`，不要绕开。

---

## 六、典型 Agent 调用顺序（一轮对话）

```
学生说话
   │
   ▼
1. GET  /students/{sid}/state                  ← 拿画像
2. GET  /students/{sid}/recall?query=学生发言   ← 拿相关历史
   │
   ▼
3. 用 state + recall 拼 prompt → 调 LLM → 得到 agent 回复
   │
   ▼
4. POST /students/{sid}/dialog                 ← 把这一轮 turns 存进去
5. POST /students/{sid}/events (×N)            ← 把你从学生发言里抽到的结构化事件写进去
                                                  (mastery_update / error_observed / preference_set ...)
   │
   ▼
返回 agent 回复给学生
```

会话级别还要做一件事：学生点退出时，发一条 `session_end` 事件。

---

## 七、不想走 HTTP？也可以直接当库用

如果你的 agent 就是个 Python 服务、跟 memory module 在同一进程，可以跳过 HTTP：

```python
from memory_module.service import build_default_service
from memory_module import schemas as S

svc = build_default_service(db_path="./memory.db")

# 写事件
svc.handle_event("alice", S.MasteryUpdateEvent(concept_id="loop", score=0.7))

# 写对话
svc.handle_dialog("alice", [
    S.DialogTurn(role="student", content="我想让小猫跑"),
    S.DialogTurn(role="agent",   content="可以用 forever"),
])

# 读
state = svc.get_state("alice")
hits  = svc.recall("alice", query="小猫", kind="dialog", top_k=3)
```

直接调函数，省一次 HTTP。建议起步阶段就这么用，等 agent 拆服务再切到 HTTP。

参考 `langgraph_demo.py` 和 `chat_cli.py` 看完整的「LLM 抽事件 → 写 → 下轮读」闭环。

---

## 八、错误处理

| 状态码 | 含义 | 你该怎么办 |
|---|---|---|
| 200 | 成功 | — |
| 400 | 请求格式不对 / `concept_id` 不在白名单 / `top_k` 越界 | 看 `detail` 字段，修正后重试 |
| 422 | Pydantic 校验不过（缺字段、类型错） | 看返回里的字段路径，对照本文修正 |
| 5xx | 服务端炸了 | 重试，连续失败的话查 stdout 日志 |

**重要：写入失败不要静默吞掉**——记忆丢一条 → 下次召回就有空洞 → agent 表现下降但没人知道哪儿出的问题。

---

## 九、限制 / 暂时不做

- 没有认证、没有限流。**不要直接暴露到公网**，挂在内网或加个网关。
- 没有删除接口（删学生、删某条记忆）。临时清理用 `chat_cli.py` 的 `/clear` 或直接 `DELETE FROM ... WHERE student_id=?`。
- 默认 embedder 是哈希向量，**召回质量不如真模型**。上线前换掉（见 README 的「升级路径」）。
- 对话摘要默认是 stub（拼接学生发言）。生产请配 LLM summarizer。
- 短期上下文管理已作为**独立功能**提供（见「三·补」：缓冲 + 按预算窗口化）；但**基于 LLM 的溢出自动摘要压缩**仍未接入——已留好 `Compactor` 接口，默认只按时间新近度裁剪。

---

## 十、联系 / 排查

- 看 stdout 日志（`logging` INFO 级别，关键路径都有）
- 跑 `uv run python demo.py` 看是否所有断言通过——通过就说明服务端没坏，问题在调用方
- 真出问题：把日志 + 复现 curl 发过来
