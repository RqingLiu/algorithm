# Employee Assistant：FastAPI + LangGraph 最小 RAG 后端

三个 API：`POST /ingest`、`POST /query`、`GET /health`。
采用 **Router → Service → Repository** 三层：HTTP、业务规则、数据访问分别负责。
Model（数据结构）与 Config（配置）是辅助模块，不是额外的串行调用层。

## 目录与职责

```text
employee-assistant/
├── app/
│   ├── main.py                  # 创建 app、生命周期、注册 Router
│   ├── config.py                # 环境变量和默认配置
│   ├── models.py                # Pydantic 请求/响应模型、LangGraph State
│   ├── errors.py                # 不依赖 HTTP 的业务异常
│   ├── providers.py             # embedding / 生成模型适配器
│   ├── routers/rag.py           # HTTP 输入、依赖注入、状态码
│   ├── services/assistant.py    # 入库版本规则、权限、会话、业务编排
│   ├── services/graph.py        # LangGraph 检索、缓存和生成节点
│   └── repositories/policies.py # JSON 原子读写、索引、缓存、检索
├── tests/                       # HTTP、权限、更新和失败原子性测试
├── examples/ingest.json          # 可直接调用的入库请求
├── policies.json                # 虚构政策样例；可手动编辑
├── main.py                      # 兼容旧启动命令的入口
├── demo.py                      # 终端观察节点和缓存
├── WALKTHROUGH.md               # 学习笔记
└── architecture.md              # 数据流程图
```

Router 不直接访问文件或检索库；Service/Repository 不导入 FastAPI。
LangGraph 是 Service 层的内部编排方式。

## 安装与启动

需要 Python 3.10+，已在 Python 3.12 上验证。进入本项目目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8081
```

已有 `.venv` 时无需重新安装。`python -m uvicorn main:app ...` 仍兼容。
打开 <http://localhost:8081/docs>；停止使用 Ctrl+C。改 Python 后重启，或开发时加 `--reload`。

默认数据文件是项目根目录 `policies.json`。如希望从空库开始并保留示例：

```bash
export POLICIES_PATH=./local-data/policies.json
python -m uvicorn app.main:app --host 127.0.0.1 --port 8081
```

不存在的数据文件被当作空库，首次入库时创建。文件用于持久化文档；向量索引与缓存在进程内存中，重启重建索引、清空缓存。
仅支持单 worker、单进程写入，不要使用多个进程同时写同一个 JSON 文件。

## API 实验

### 1. GET /health

```bash
curl http://localhost:8081/health
```

返回 `status`、`mode`、文档数、chunk 数、索引版本。
这是进程与已加载索引的健康信息，不会调用模型检查 Ollama 是否在线，也不会在健康检查时重新读取文件。

### 2. POST /ingest

```bash
curl -X POST http://localhost:8081/ingest \
  -H 'Content-Type: application/json' \
  -H 'X-Demo-User: helen' \
  --data-binary @examples/ingest.json
```

请求为 JSON 文档列表，不是 multipart PDF 上传。支持纯文本或已解析 Markdown；不包含 PDF/OCR。
每批最多 20 篇，每篇正文最多 100000 字符。分块按二级标题，长块按中文句子和字符兜底（400 字符，不是 token）。

- 新 id：自动 `version=1`。
- 同 id 且规范化内容相同：不重复升级版本，`changed=false`。
- 同 id 内容或权限等字段改变：自动递增版本并替换该文档。
- 不传 `version`，传入会被拒绝（422）。同一批次重复 id 也拒绝。
- 先构建整批索引，成功后原子替换文件，再发布内存快照；失败不发布部分数据。
- 版本号会保存，但暂不保留历史修订全文。手动编辑文件不会自动增加其中的 version。
- 更新索引版本后，旧答案/检索缓存的 key 不再匹配；未变化的编码文本复用 embedding。

演示用户 helen 可入库，alice/bob 为只读。未知用户返回 401，无写入权限返回 403，模型/存储失败返回 503。

### 3. POST /query

```bash
curl -X POST http://localhost:8081/query \
  -H 'Content-Type: application/json' \
  -H 'X-Demo-User: alice' \
  -d '{"question":"上海员工去北京出差，住宿上限是多少？","session_id":"lesson-1"}'
```

继续同一会话：

```bash
curl -X POST http://localhost:8081/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"那深圳呢？","session_id":"lesson-1"}'
```

查看 `standalone_query`（改写结果）、`trace`（节点/缓存轨迹）、`citations`（证据与版本）、`answer`。
在 Swagger 中使用 `/ingest` 时，将 x-demo-user 改为 helen。

## 核心数据流程

```text
POST /ingest
  → Router 校验请求
  → Service 检查写入权限、按 id 分配版本
  → Repository 分块、复用/生成 embedding、原子保存 JSON、发布索引
  → 返回文档版本和索引版本

POST /query
  → Router
  → Service 刷新索引、读取用户/会话状态
  → LangGraph 改写 → 构造缓存 key → 查答案/检索/向量缓存
  → Repository 权限/地区/有效期过滤 → BM25 + 向量检索 → RRF
  → 教学重排 → 回答 → 缓存 → 响应
```

## 用户、缓存与多轮

| 用户 | 地区 | 用户组 |
| --- | --- | --- |
| alice | 上海 | employees |
| helen | 上海 | employees、hr |
| bob | 纽约 | employees |

`X-Demo-User` 是本地演示身份选择器，不是真实认证。只监听 localhost，不要原样部署到企业网络。
权限来自 Service 的用户表；文档 ACL 来自 groups 字段。检索前两者必须有交集。
有效期为 `[effective_from, effective_to)`，只允许 published 状态。

三个在线缓存默认 TTL 300 秒；会话 TTL 3600 秒；文档向量缓存 TTL 86400 秒。
缓存用内存字典，写满时逐出最早插入记录，非 LRU。不同用户的答案/检索结果保守隔离；相同查询文本可复用向量。

运行 `python demo.py`，观察重复年假问题命中答案缓存；bob 提相同问题会重新检索纽约政策，但复用查询向量。
多轮为有限规则改写，目前覆盖住宿、年假、薪酬及北京/深圳目的地追问；不是通用自然语言理解。

## 接真实模型（可选）

默认 `MODEL_MODE=demo`：特征哈希向量 + 原文摘录，**不是语义 embedding，也不是 LLM 生成**。
BM25、RRF、LangGraph 与 API 均真实执行；向量存储是内存，没有外部向量数据库。
重排为词项覆盖率教学实现，不是 Cross-Encoder。

代码已实现 Ollama `/api/embed` 与 `/api/generate` 接口。先安装并下载自己选定的本地模型，再设置：

```bash
export MODEL_MODE=ollama
export EMBED_MODEL='已安装的embedding模型名称'
export CHAT_MODEL='已安装的聊天模型名称'
export OLLAMA_URL=http://localhost:11434
python -m uvicorn app.main:app --host 127.0.0.1 --port 8081
```

配置集中在 `app/config.py`；`.env.example` 供参考，不会自动加载 `.env`。
真实 Ollama 推理未在本轮测试；测试全部使用无网络 demo 模式。
模型失败会报错，不静默降级。模型标签需固定版本，更换后重启。

## 测试与格式检查

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

测试使用临时文件和独立 fixture，不受手动修改 policies.json 金额影响。
验证入库/查询/重启持久化、版本幂等、权限与缓存隔离、过期政策、多轮、模型及磁盘失败回滚。
requirements-lock.txt 为最初运行环境的完整依赖快照；requirements.txt 为固定的直接运行依赖。

## 当前边界

这是单进程教学最小版：没有 SSO、SharePoint 同步、PDF/OCR、Redis、数据库事务和分布式向量索引。
没有历史版本审计存储；没有删除 API，可通过入库更新 status=inactive 停用文档。
无端到端事实核验：找到候选不等于证据完整；citations 是交给生成器的资料，不保证模型每句话都有依据。
所有样例政策和用户均为虚构，不用于真实 HR 决策。

## 参考

- [FastAPI 多文件与 APIRouter](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [官方 retrieval-agent-template](https://github.com/langchain-ai/retrieval-agent-template)
- [官方 rag-research-agent-template](https://github.com/langchain-ai/rag-research-agent-template)：2026-03-11 已归档，仅参考架构。
- [Ollama embedding](https://docs.ollama.com/api/embed) / [生成](https://docs.ollama.com/api/generate)
