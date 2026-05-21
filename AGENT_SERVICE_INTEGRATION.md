# Agent 调用记忆服务指示文档

这份文档只讲一件事：

你的 agent 怎样通过 HTTP 调用当前这个记忆模块服务。

重点覆盖两类能力：

1. 结构化长期记忆
2. agent 诊断记忆与服务端情感判断

---

## 1. 服务启动

在仓库根目录启动：

```bash
uv sync
uv run uvicorn memory_module.api:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

预期返回：

```json
{"status":"ok"}
```

如果你要启用服务端 MiMo 情感判断，确保环境里有：

```bash
MEMORY_MODULE_USE_MIMO=1
MIMO_API_KEY=...
```

说明：

- 开了 `MEMORY_MODULE_USE_MIMO=1` 后，服务端会对对话摘要和迷思聚类启用 MiMo。
- 对 `POST /agent-turns` 写入的 agent 回合，服务端会自动做情感判断。
- 如果 MiMo 不可用，服务端会自动回退到规则判断。

---

## 2. 你应该怎么理解 `student_id`

所有接口都按 `student_id` 隔离。

建议：

- `student_id` 用稳定用户标识
- `session_id` 用一次会话标识
- `task_id` 用一道题 / 一个任务 / 一个项目的稳定标识

推荐关系：

```text
student_id = 用户本身
session_id = 本次会话
task_id    = 当前任务
```

不要把 `student_id` 直接做成一次性 session 值，否则跨 session 记忆会断。

---

## 3. Agent 最常用的 4 个接口

### 3.1 读 agent 决策快照

```http
GET /students/{student_id}/agent-memory?session_id=...&task_id=...
```

用途：

- 在生成下一句反馈之前，读取这个学生最近的错误类型、反馈等级、薄弱概念、情感状态

示例：

```bash
curl "http://127.0.0.1:8000/students/alice/agent-memory?session_id=s2&task_id=loop-cat"
```

返回示例：

```json
{
  "recent_feedbacks": ["先看 forever 放的位置"],
  "recent_error_types": ["WRONG_STRUCTURE"],
  "recent_feedback_level": 2,
  "repeated_error_count": 1,
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

你应该重点消费这些字段：

- `recent_error_types`
- `recent_feedback_level`
- `repeated_error_count`
- `weak_concepts`
- `affective_state`

---

### 3.2 写一次 agent 诊断回合

```http
POST /students/{student_id}/agent-turns
```

这是 agent 侧最重要的写接口。

服务端收到这条记录后会自动：

1. 记录最近错误和反馈等级
2. 推断薄弱概念
3. 自动生成一条情感事件
4. 在启用 MiMo 时优先走服务端 LLM 情感判断

所以通常情况下：

**你只需要写 `agent-turns`，不需要再单独写 `/affective-events`。**

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

返回：

```json
{"turn_id": 12}
```

关键字段说明：

- `passed`: 这次是否通过
- `score`: 0 到 1
- `primary_error_type`: 本轮主要错误类型
- `concept_id`: 可选，但强烈建议传
- `feedback_level`: 1 到 3
- `feedback_text`: 你已经给学生说出去的话
- `user_text`: 学生原话，**建议尽量传**
- `diff_summary`: 对本轮错误的简洁描述
- `highlight_node_ids`: 前端高亮节点 id

其中 `user_text` 很重要：

- 如果你想让服务端 LLM 更准确地判断 frustration / confusion / confidence
- 就必须把学生原话一起传上来

如果不传 `user_text`，服务端仍然能判断情感，但更多依赖行为信号，而不是文本信号。

---

### 3.3 写普通对话历史

```http
POST /students/{student_id}/dialog
```

这个接口用于存储普通对话流，服务端会自动摘要并做召回索引。

示例：

```json
{
  "turns": [
    {"role": "student", "content": "我想让小猫一直跑"},
    {"role": "agent", "content": "可以试试 forever 积木"}
  ]
}
```

返回：

```json
{"turn_id": 7}
```

---

### 3.4 语义召回

```http
GET /students/{student_id}/recall?query=...&kind=dialog&top_k=5
```

`kind` 有两个值：

- `dialog`
- `misconception`

用途：

- 学生提到“上次那个猫的项目”时，召回历史对话
- 学生反复卡在同类错误上时，召回历史迷思

---

## 4. 推荐调用顺序

如果你的 agent 是一次一评测、一反馈的模式，推荐每轮这样调：

### 回合开始

1. 调 `GET /students/{student_id}/agent-memory`
2. 可选调 `GET /students/{student_id}/state`
3. 可选调 `GET /students/{student_id}/recall`
4. 用这些结果构造 prompt

### 回合结束

1. 先把师生对话写入 `POST /dialog`
2. 再把本轮诊断结果写入 `POST /agent-turns`
3. 如果有掌握度、偏好、项目保存等结构化信息，再写 `POST /events`

推荐顺序图：

```text
load agent-memory
  -> load state / recall
  -> agent 生成反馈
  -> POST /dialog
  -> POST /agent-turns
  -> POST /events (可选)
```

---

## 5. Python 最小调用示例

```python
import requests

BASE = "http://127.0.0.1:8000"
student_id = "alice"
session_id = "s2"
task_id = "loop-cat"

# 1. 读 agent 决策快照
snap = requests.get(
    f"{BASE}/students/{student_id}/agent-memory",
    params={"session_id": session_id, "task_id": task_id},
    timeout=5,
).json()

# 2. 生成反馈后，写本轮 agent turn
payload = {
    "session_id": session_id,
    "task_id": task_id,
    "attempt_id": "a-002",
    "project_name": "loop-cat",
    "passed": False,
    "score": 0.35,
    "primary_error_type": "WRONG_STRUCTURE",
    "concept_id": "loop",
    "feedback_level": 2,
    "feedback_text": "先找 forever 积木，再看运动积木是不是放进去了。",
    "user_text": "我还是不会，这个太难了",
    "diff_summary": "missing forever block around motion",
    "highlight_node_ids": ["block-1"],
    "need_rag": False,
    "need_repair_validation": False,
}

resp = requests.post(
    f"{BASE}/students/{student_id}/agent-turns",
    json=payload,
    timeout=5,
)
resp.raise_for_status()
print(resp.json())
```

---

## 6. 如果你还要写结构化教学记忆

除了 `agent-turns`，你还可以继续写：

```http
POST /students/{student_id}/events
```

常见事件：

- `mastery_update`
- `session_end`
- `project_saved`
- `preference_set`
- `error_observed`

示例：

```json
{
  "event": {
    "type": "mastery_update",
    "concept_id": "loop",
    "score": 0.7,
    "evidence_note": "独立用 forever 完成动画"
  }
}
```

---

## 7. 什么时候才需要 `/affective-events`

接口：

```http
POST /students/{student_id}/affective-events
```

正常情况下你不需要主动调它。

只在这些场景才建议调用：

- 你的前端单独采集到了情绪按钮/量表
- 你有外部模型专门做情感识别
- 你想人工覆盖服务端自动判断

否则只写 `/agent-turns` 就够了。

---

## 8. 概念白名单

`concept_id` 只能从这 12 个里选：

```text
loop  conditional  event  variable  broadcast  clone
sprite  sound  motion  sensing  operator  list
```

传错会返回 400。

---

## 9. 错误处理建议

常见状态码：

- `200`: 成功
- `400`: 字段值非法，比如 `concept_id` 错
- `422`: 请求体结构不对
- `5xx`: 服务端异常

建议：

- 所有写接口都打日志
- 写失败不要静默吞掉
- 至少对 `/agent-turns` 做重试或告警

---

## 10. 接入时最容易漏掉的点

### 漏点 1：没有传 `user_text`

后果：

- 服务端情感判断少了文本信号
- frustration / confusion 的估计会变保守

### 漏点 2：把 `student_id` 当 `session_id`

后果：

- 新 session 读不到历史
- 长期记忆退化成单次会话缓存

### 漏点 3：写了 `dialog` 但没写 `agent-turns`

后果：

- 普通对话可召回
- 但 `recent_feedback_level / weak_concepts / affective_state` 不会更新

### 漏点 4：已经写了 `agent-turns`，又重复写 `/affective-events`

后果：

- 情感状态会被双写
- 聚合结果可能偏高

除非你是明确做人工覆盖，否则不要重复写。

---

## 11. 建议你在 agent 里至少实现这两个函数

### `load_memory_snapshot(student_id, session_id, task_id)`

内部调用：

```http
GET /students/{student_id}/agent-memory
```

### `persist_agent_turn(student_id, payload)`

内部调用：

```http
POST /students/{student_id}/agent-turns
```

如果你的 agent 先把这两个函数接好，整条记忆闭环就跑起来了。

---

## 12. 当前服务端关于情感判断的实际行为

当前实现是：

- `POST /agent-turns` 时自动触发
- 优先尝试 MiMo LLM 判断
- MiMo 不可用时回退规则
- 聚合后通过 `GET /agent-memory` 返回

也就是说：

**agent 不需要自己做最终情感判断。**

agent 只要把事实和学生原话交给服务端。

