# 原型实际数据流程

## 入库：启动时与每次请求前检查变更

```mermaid
flowchart LR
    P[虚构政策 JSON / 已解析 Markdown] --> H[内容 hash / 索引版本]
    H --> S[按标题分块 / 长块句子与字符兜底]
    S --> E[文档 embedding 缓存 / 模型编码]
    E --> I[内存 chunk + vector + ACL 元数据快照]
```

## 在线：图中的节点名与代码一致

```mermaid
flowchart TD
    HTTP[POST /query] --> S[Service: 用户与会话 / 刷新索引]
    S --> rewrite
    rewrite -->|不明确或不支持| refuse
    rewrite --> keys
    keys --> answer_cache
    answer_cache -->|HIT| END
    answer_cache -->|MISS| retrieval_cache
    retrieval_cache -->|HIT 且有证据| generate
    retrieval_cache -->|HIT 但空结果| refuse
    retrieval_cache -->|MISS| embed
    embed --> retrieve
    retrieve --> rerank
    rerank -->|有候选| generate
    rerank -->|无候选| refuse
    generate --> END
    refuse --> END
```

`embed` 节点内部先查向量缓存；`retrieve` 节点内部先做 ACL/地区/有效期过滤。
重排结果写检索缓存，生成结果写答案缓存，Service 最后保存结构化会话状态。
本图没有把未实现的 SharePoint/OCR/SSO/Vector DB 画成已经存在的模块。


## 新增：HTTP 入库路径

```mermaid
flowchart LR
    A[POST /ingest JSON] --> B[Router: Pydantic 校验]
    B --> C[Service: 写入权限 / 比较文档 / 自动版本]
    C --> D[Repository: 分块 / embedding / 构建快照]
    D --> E[原子替换 JSON 文件]
    E --> F[发布内存索引版本]
    F --> G[返回版本与 chunk 数]
```

GET /health 经 Router 调用 Service，读取已加载索引统计，不调用外部模型。
