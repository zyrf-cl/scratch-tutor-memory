# Agent 接入指南

给教学 agent 开发者：怎么把这个记忆模块接进你的 agent。

如果你只想跑通看看，直接跳到第 1 节。如果你要写代码，从第 3 节开始。

---

## 0. 这是什么

一个独立的「学生长期记忆」服务，给 Scratch 编程教学 agent 用。提供 6 类记忆：

| 通道 | 内容 | 一句话用途 |
|---|---|---|
| A 掌握度 | concept → score | 知道学生哪些概念会、哪些不会 |
| B 会话续点 | last project / pending task | 续课时一句"上次小猫飞出屏幕，今天接着改"就接上了 |
| C 对话历史 | 摘要 + 向量 | 学生说"那个动画"时能找回历史话题 |
| D 项目档案 | title / structure / highlights / issues | 学生作品的元信息 |
| E 偏好 | key → value | 学生喜欢猫还是恐龙 |
| F 错误模式 | 聚类后的迷思 | 反复栽同一坑要换种讲法 |

外加 agent 侧的运行状态：`recent_feedback_level`、`weak_concepts`、`affective_state`（沮丧/困惑度）—— 这部分让 agent 跨 session 决策时知道该用哪一级提示、要不要先安抚情绪。

**学生之间完全隔离**。默认 SQLite 持久化，也可以切到 Milvus 后端，开源 MIT。

---

## 1. 5 分钟跑通验证

```bash
git clone https://github.com/zyrf-cl/scratch-tutor-memory.git
cd scratch-tutor-memory
uv sync
uv run python demo.py
```

应当打印一长串 INFO 日志然后 `演示结束。所有断言通过。`。如果通过，说明这台机器上模块本身一切正常。

想看 LLM 真的接进来：

```bash
cp .env.example .env       # 填进自己的 MIMO_API_KEY
uv run python langgraph_demo.py
```

---

## 2. 接入方式选哪个

| 你的处境 | 选 |
|---|---|
| Agent 是 Python，跟记忆模块跑同一进程 | **库模式**（推荐起步） |
| Agent 是别的语言 / 跟记忆模块跨服务 | **HTTP 模式** |
| 起步阶段还在验证流程 | **库模式**，省一次 HTTP 跳跃，调试也容易 |
| 要部署到生产 / 多个 agent 实例共享一份记忆 | **HTTP 模式** |

两种模式底层是同一套代码，行为完全一致。可以先用库模式跑通，再切 HTTP，不需要改业务逻辑。

---

## 3. 库模式接入（6 行最小例子）

```python
from memory_module.service import build_default_service
from memory_module import schemas as S

svc = build_default_service(db_path="./memory.db")

# 写一条掌握度事件
svc.handle_event("alice", S.MasteryUpdateEvent(concept_id="loop", score=0.7))

# 读完整画像
state = svc.get_state("alice")
print(state.mastery)
```

就这样。`MemoryService` 是你需要的所有东西，方法表见 3.3 节。

### 3.1 一轮对话的标准调用顺序

```
学生说话
  ↓
1. svc.get_state(sid)                          ← 拿画像
2. svc.recall(sid, query=学生发言, kind=...)    ← 拿相关历史
  ↓
3. 用 state + recall 拼 prompt → 调 LLM → 得到 agent 回复
  ↓
4. svc.handle_dialog(sid, [DialogTurn(...)])   ← 把这一轮 turns 存进去
5. svc.handle_event(sid, MasteryUpdateEvent(...))    ← 从学生发言抽到的结构化事件
   svc.handle_event(sid, ErrorObservedEvent(...))     （可能多条）
  ↓
返回 agent 回复给学生
```

会话级别还要一件事：学生退出时发一条 `SessionEndEvent`：

```python
svc.handle_event("alice", S.SessionEndEvent(
    current_project_id="proj-1",
    current_topic="动画",
    pending_task="下次解决小猫飞出屏幕",   # ← 续课关键
))
```

### 3.2 PracticeAgent 风格的接入（带 feedback_level 升级）

如果你的 agent 像 PracticeAgent 那样有"一次答题尝试"这种语义（错误类型、提示级别、情感推断），调用 `handle_agent_turn` + `get_agent_memory_snapshot` 这一对：

```python
# 写一次尝试
svc.handle_agent_turn("alice", S.AgentTurnWrite(
    session_id="s1",
    task_id="loop-cat",
    project_name="proj",
    passed=False,
    score=0.4,
    primary_error_type="WRONG_STRUCTURE",
    concept_id="loop",
    feedback_level=1,
    feedback_text="...",
    diff_summary="missing forever block",
    highlight_node_ids=["block-1"],
))

# 下一次（甚至下一个 session）开始时读 snapshot
snap = svc.get_agent_memory_snapshot("alice", session_id="s2", task_id="loop-cat")
# snap.recent_error_types  → ['WRONG_STRUCTURE']
# snap.weak_concepts       → ['loop']（连续犯错聚类）
# snap.affective_state     → {'frustration': ..., 'confusion': ...}
```

`snap` 就是 agent 写下一句话前需要看到的全部历史。直接喂给 LLM prompt。

可选：写情感信号

```python
svc.handle_affective_event("alice", S.AffectiveEventWrite(
    session_id="s2", task_id="loop-cat",
    frustration_score=0.8, confusion_score=0.7,
    source="agent_inferred",
    evidence="连续两次同类错误",
))
```

完整的可运行例子见 `demo_agent.py`（含 LLM 对照实验：同一个错误，记忆不同 → LLM 输出不同）。

### 3.3 `MemoryService` 方法总览

| 方法 | 输入 | 用途 |
|---|---|---|
| `handle_event(sid, event)` | 5 种 Event 之一 | 写结构化事件（掌握度/会话结束/项目存档/偏好/错误观察） |
| `handle_dialog(sid, turns)` | `list[DialogTurn]` | 写一轮对话；服务端自动摘要 + 向量化 |
| `get_state(sid)` | — | 完整画像（6 通道全量） |
| `recall(sid, query, kind, top_k)` | query 字符串 | 语义检索 `kind="dialog"` or `"misconception"` |
| `handle_agent_turn(sid, turn)` | `AgentTurnWrite` | 写一次答题尝试 |
| `get_agent_memory_snapshot(sid, ...)` | — | 拿 agent 决策需要的 snapshot |
| `handle_affective_event(sid, ev)` | `AffectiveEventWrite` | 写一条情感信号 |

---

## 4. HTTP 模式接入

```bash
uv run uvicorn memory_module.api:app --port 8000
```

4 个端点，全部按 `student_id` 分片：

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/students/{sid}/events` | 写结构化事件，body = `{"event": {...}}` |
| POST | `/students/{sid}/dialog` | 写对话流，body = `{"turns": [...]}` |
| GET  | `/students/{sid}/state` | 拿完整画像 |
| GET  | `/students/{sid}/recall?query=...&kind=dialog&top_k=5` | 语义检索 |

Swagger 文档：`http://127.0.0.1:8000/docs`。详细端点契约见 `INTEGRATION.md`。

---

## 5. 五种 Event 速查

每种事件都包成 `{"event": <下面>}` 发到 events 端点；库模式直接传 Pydantic 对象。

```python
# A 掌握度（concept_id 必须在白名单内）
S.MasteryUpdateEvent(concept_id="loop", score=0.7, evidence_note="独立完成动画")

# B 会话结束（三字段都可选，pending_task 强烈建议填）
S.SessionEndEvent(current_project_id="proj-1", pending_task="...")

# D 项目存档
S.ProjectSavedEvent(project_id="proj-1", title="跑步小猫",
                    highlights=["首次用 forever"], issues=["小猫飞出屏幕"])

# E 偏好（source: "self" 或 "inferred"）
S.PreferenceSetEvent(key="topic", value="动画", source="self")

# F 出错观察（concept_id 可选但强烈建议）
S.ErrorObservedEvent(concept_id="event", description="把 when-flag-clicked 当 forever 用了")
```

**概念白名单**（concept_id 字段只能从这 12 个里选）：

```
loop  conditional  event  variable  broadcast  clone
sprite  sound  motion  sensing  operator  list
```

发别的字符串会被服务端拒绝。要增减找模块维护者改 `memory_module/concepts.py`。

---

## 6. 验证你的接入是否正确

按从弱到强排：

| 级别 | 跑什么 | 证明了什么 |
|---|---|---|
| L1 | `uv run python demo.py` | 模块本身没坏 |
| L2 | `uv run python langgraph_demo.py` | LLM extract→write→recall 闭环对 |
| L3 | `uv run python chat_cli.py --student alice`，自己聊几轮 | 体感 + SQLite 文件里能看到记忆 |
| L4 | 你自己的 agent 跑一遍，断言 snapshot 字段符合预期 | 你接得对 |

写 L4 断言的模板见 `demo_agent.py` 末尾那一段（"自检断言"），可以直接抄。

---

## 7. 常见坑

**1. concept_id 写错** → 400 错误。先核对白名单。

**2. 写入失败不要静默吞掉** → 记忆丢一条，下次召回有空洞，agent 表现下降但无人知晓。在调用方加日志或重试。

**3. 默认 embedder 行为** → 开箱即用时，`Embedder` 默认是 `bge`（真 sentence-transformers 模型），但用 `FallbackEmbedder` 包了一层 —— 没装 `sentence-transformers` extra 时会**静默退到 hash**。要看到退化日志或强制要求真模型，见第 8 节。

**4. 对话摘要 / 迷思聚类的 LLM 实现已写好但默认关闭** → 设 `MEMORY_MODULE_USE_MIMO=1` + `MIMO_API_KEY` 才会启用。详见第 8 节。

**5. 没有认证、没有限流** → HTTP 模式不要直接暴露公网。挂内网或加网关。

**6. MiMo / 其他 reasoning 模型 `max_tokens` 太小** → 模型把 token 全用在思考上，正式输出空。给到 2048 以上比较稳。

**7. 没有删除接口** → 临时清理用 `chat_cli.py` 的 `/clear` 或直接 `DELETE FROM ... WHERE student_id=?`。

---

## 8. 升级到生产实现

模块里 `Embedder` / `DialogSummarizer` / `MisconceptionDetector` 都是 Protocol，**Stub 实现和真实现都已经写好了**，按下面切换即可。

### 8.1 升级一览

| 接口 | 默认行为 | 启用真实现 |
|---|---|---|
| `Embedder` | `bge` 配 hash 退化 | `uv sync --extra st` 装上 sentence-transformers |
| `DialogSummarizer` | `StubDialogSummarizer`（拼接学生发言） | `MEMORY_MODULE_USE_MIMO=1` + `MIMO_API_KEY` |
| `MisconceptionDetector` | `StubMisconceptionDetector`（按 concept 计数） | 同上 |
| `MemoryStore` | `SQLiteStore`（本地文件） | `MEMORY_MODULE_STORE=milvus` + `uv sync --extra milvus` |

### 8.2 完整环境变量表

| 变量 | 默认 | 作用 |
|---|---|---|
| `MEMORY_MODULE_DB` | `./memory.db` | SQLite 路径；`:memory:` = 纯内存 |
| `MEMORY_MODULE_STORE` | `sqlite` | `sqlite` / `milvus` |
| `MEMORY_MODULE_EMBEDDER` | `bge` | `bge` / `hash` / `stub` |
| `MEMORY_MODULE_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 任何 sentence-transformers 模型名 |
| `MEMORY_MODULE_EMBEDDING_DEVICE` | 自动 | `cuda` / `cpu` / `mps` |
| `MEMORY_MODULE_EMBEDDING_STRICT` | `0` | 设 `1` 时缺依赖直接报错，不退化 |
| `MEMORY_MODULE_USE_MIMO` | `0` | 设 `1` 启用 MiMo 摘要 + 迷思聚类 |
| `MIMO_API_KEY` | — | MiMo 必填 |
| `MIMO_BASE_URL` | `https://token-plan-cn.xiaomimimo.com/v1` | 改 endpoint |
| `MIMO_MODEL` | `mimo-v2.5-pro` | 改模型名 |
| `MEMORY_MODULE_MILVUS_URI` | `http://localhost:19530` | Milvus 地址 |
| `MEMORY_MODULE_MILVUS_TOKEN` | — | Milvus token 或 `user:password` |
| `MEMORY_MODULE_MILVUS_DB` | — | Milvus database 名 |
| `MEMORY_MODULE_MILVUS_PREFIX` | `memory_module` | collection 前缀 |
| `MEMORY_MODULE_MILVUS_TIMEOUT` | `5` | Milvus 操作超时秒数 |
| `MEMORY_MODULE_MILVUS_CONSISTENCY` | `Strong` | Milvus 一致性级别 |

### 8.3 实战命令组合

**全量生产配置**（真 embedding + 真 LLM 摘要 + 真聚类）：

```bash
uv sync --extra st
export MEMORY_MODULE_USE_MIMO=1
uv run --env-file .env python demo.py
```

**只换 embedder，不上 LLM**（向量检索质量更高，但摘要还走拼接）：

```bash
uv sync --extra st
uv run python demo.py
```

**调试用，强制要求真模型（缺依赖直接炸）**：

```bash
MEMORY_MODULE_EMBEDDING_STRICT=1 uv run python demo.py
```

**换更大的中文 BGE**：

```bash
MEMORY_MODULE_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5 uv run python demo.py
```

**切到 Milvus 后端**（需要本机或远端 Milvus 服务）：

```bash
uv sync --extra milvus
MEMORY_MODULE_STORE=milvus \
MEMORY_MODULE_MILVUS_URI=http://localhost:19530 \
MEMORY_MODULE_EMBEDDER=hash \
uv run python scripts/smoke_milvus_store.py
```

Milvus 主要提升大规模向量召回、并发和部署扩展性；小规模 demo 用 SQLite 更简单。召回质量本身仍取决于 embedding 模型和摘要质量，不是换库自动变聪明。

### 8.4 接你自己的实现（OpenAI / Claude / Cohere / 自建服务）

三个 Protocol 都是鸭子类型，**不用继承类、不用 fork 代码**。直接写一个有正确方法的对象塞给 `MemoryService`：

```python
import numpy as np
from memory_module.service import MemoryService
from memory_module.store import SQLiteStore

class OpenAIEmbedder:
    @property
    def dim(self) -> int:
        return 1536

    def embed(self, text: str) -> np.ndarray:
        # 调你的 OpenAI embedding API
        vec = openai_client.embeddings.create(
            input=text, model="text-embedding-3-small",
        ).data[0].embedding
        v = np.asarray(vec, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)   # 务必 L2 归一化


class ClaudeSummarizer:
    def summarize(self, turns: list) -> str:
        # turns: list[DialogTurn(role, content)]
        transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": f"给出对话一句话摘要：\n{transcript}"}],
        )
        return resp.content[0].text


class MyDetector:
    def detect(self, errors: list[dict]):
        # 返回 list[MisconceptionCluster]，至少要有 pattern_id / description / concept_ids
        ...

svc = MemoryService(
    store=SQLiteStore(db_path="./memory.db"),
    embedder=OpenAIEmbedder(),
    dialog_summarizer=ClaudeSummarizer(),
    misconception_detector=MyDetector(),
)
```

写完后跑一次 `demo.py` 把 store 路径指到内存模式，看断言是否通过 —— 通过就说明接口契约对了。

Protocol 定义见 `memory_module/embedding.py` 的 `Embedder` 和 `memory_module/extractor.py` 的 `DialogSummarizer` / `MisconceptionDetector`。

---

## 9. 不在 v1 范围内

- LLM 上下文管理（你 agent 自己决定把 state 哪些字段塞进 prompt）
- 认证、限流、家长后台、班级聚合
- 真正的教学 agent（demo 用模拟脚本喂事件）

---

## 10. 排查

- 看 stdout 日志（INFO 级别，关键路径都打）
- 跑 `demo.py` 看是否所有断言通过 —— 通过就说明服务端没坏，问题在调用方
- 真出问题：把日志 + 复现脚本发过来
