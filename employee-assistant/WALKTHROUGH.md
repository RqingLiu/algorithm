> 三层重构说明：当前 Router 位于 app/routers/rag.py，main.py 仅兼容旧启动命令。新增 /ingest、/health 和自动入库版本，完整使用方法见 README.md。

# 一起读代码：先理解一条请求如何走完

先运行 `python demo.py`，再按下面顺序读。无需一口气理解所有文件。

## 第一课：FastAPI 与 LangGraph 各管什么

`app/routers/rag.py` 的 `query()` 接收 HTTP 参数，调用 `Assistant.ask()`。
`app/services/assistant.py` 中的 Assistant 维护依赖和会话，再调用 `self.graph.invoke(...)`。
`app/services/graph.py` 把业务流程拆成节点，决定下一步去哪。

```text
HTTP 请求 → Router → Service → LangGraph 工作流 → Data / 模型适配器
```

LangGraph 不替代 FastAPI，也不是向量数据库；它负责有状态的流程编排。
本例是确定性工作流，不是一个自主决定所有动作的 agent。

## 第二课：State 是什么

读 `app/models.py` 的 `State`。它是一轮图执行中共享的数据结构。

```python
question: str       # 用户这轮原话
query: str          # 补全后的独立检索问题
docs: list[dict]    # 检索结果
answer: str        # 最终回答
```

节点签名类似 `def rewrite(state: State)`：读取 state，只返回本节点要更新的字段。
例如 `return {'query': query}` 不会删除原有的 question。

`trace: Annotated[list[str], operator.add]` 的意思是：trace 使用列表相加合并更新。
每个节点返回一条日志，最后得到整个执行路径。其他字段默认覆盖更新。

State 不是跨轮历史。本例跨轮状态由 Service 按 `(user_id, session_id)` 保存，再作为下一次 invoke 的 context 输入。
故意没有先引入 checkpointer，避免把图内 State、长期会话和缓存混为一谈。

## 第三课：Node 与 Edge

读 `app/services/graph.py` 末尾：

```python
builder = StateGraph(State)
builder.add_node('rewrite', rewrite)
builder.add_edge(START, 'rewrite')
```

含义是“用 State 作为数据结构，注册 rewrite 函数，从它开始”。
`add_node` 只是登记函数，此时不执行。
`compile()` 构建可执行图，`invoke(initial_state)` 才执行一轮。

普通 edge 表示固定下一步。conditional edge 用当前 state 选择分支。
答案缓存的条件边：命中走 END，否则去 retrieval_cache。
这正是我们之前讨论的“查到更靠后的有效结果，就跳过前面的计算”。

## 第四课：Key 不等于 Value

读 `keys()`：只调用 `cache_key()` 做 SHA-256，没有调用模型。

```text
embedding_key → 查询向量
retrieval_key → 排序后的 chunks
answer_key → 答案 + 所引用 chunks
```

检索 key 包含 embedding key、权限/地区、过滤条件、索引版本、日期和搜索配置。
答案 key 再包含生成配置。不同用户的答案保守隔离，即使权限相同也不共享；查询向量可以复用。
只有走到 embed 节点且向量缓存 MISS 时，才调用模型编码。

## 第五课：多轮改写不会修改权限

第一轮含有“北京、住宿”，记录 topic、destination、original_question。
第二轮“那深圳呢？”读取同一用户同一会话的状态，替换目的地，保留原问题意图。
“需要什么材料”仍然问材料，不会被改成“上限是多少”。
这是有限规则演示。后续可以替换成 LLM 输出经过校验的结构化 QueryPlan，并保留这些测试。
用户权限始终从 Service 的演示用户表取，不从问题或历史中推断。

## 第六课：先过滤，再搜索

读 `app/repositories/policies.py` 的 `eligible()`：按用户组交集、员工地区、发布状态和有效期筛选。
然后在筛选后的候选上运行 BM25 和向量相似度，用 RRF 融合。
没有跨越权限边界后再交给模型自行判断。
真正企业数据库要把这些条件下推到索引查询，不是把所有文档加载进 Python。
po
## 建议动手实验

1. 改 `policies.json` 中的金额，验证缓存没有返回旧金额。
2. 给同一问题切换 alice / bob，看查询向量复用与地区隔离。
3. 在 graph 的某个节点添加一条 trace，运行 demo 观察它在何时出现。
4. 读懂默认流程后，再接真实 embedding 与 LLM，比较哪些节点变了、哪些没变。

不要把默认 hash 向量的检索表现当作语义 embedding 的效果，也不要把摘录模式当作 LLM 已完成答案生成。
