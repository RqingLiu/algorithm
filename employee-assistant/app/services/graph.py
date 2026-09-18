"""从这里学习 LangGraph：State 是数据，node 是函数，edge 决定下一步。"""

import re

from langgraph.graph import END, START, StateGraph

from app.models import State
from app.providers import tokens
from app.repositories.policies import cache_key


def build_graph(repo, models, caches):
    def rewrite(state: State):
        question = state["question"].strip()
        topic = next((t for t in ("住宿", "年假", "薪酬") if t in question), None)
        city = next((c for c in ("北京", "深圳") if c in question), None)
        slots = {}
        clarification = ""
        # 有限规则演示省略补全。不会把所有历史和模型回答拼进检索问题。
        if topic:
            slots = {"topic": topic, "original_question": question}
            if topic == "住宿":
                slots["destination"] = city
                if not city:
                    clarification = "请说明出差目的地：北京还是深圳？"
        elif city and re.fullmatch(r"(?:那|如果去|去)?(?:北京|深圳)(?:呢|怎么样)?[？?]?", question):
            if state["context"].get("topic") == "住宿":
                slots = {**state["context"], "destination": city}
            else:
                clarification = "你想了解这个城市的什么政策？例如住宿报销。"
        else:
            clarification = "教学版只支持住宿、年假和薪酬政策。请明确主题；其他问题需转 HR。"
        if slots.get("topic") == "年假" and any(w in question for w in ("剩", "余额", "几天")):
            clarification = "个人年假余额需要授权访问 HR 系统；本原型没有该接口，不能推算你的天数。"
        if any(w in question for w in ("去年", "旧版", "以前")):
            clarification = "本原型只检索当前政策，暂不支持历史政策查询。"
        query = question
        if slots.get("topic") == "住宿" and not clarification:
            # 保留原问题意图，例如“要什么材料”不能被重写成“上限是多少”。
            query = slots["original_question"]
            if not topic:
                previous_city = state["context"].get("destination")
                if previous_city:
                    query = query.replace(previous_city, city)
                else:
                    query += f"（目的地：{city}）"
                slots["original_question"] = query
        return {
            "query": query,
            "slots": slots,
            "clarification": clarification,
            "trace": [f"rewrite：{query}（规则改写）"],
        }

    def keys(state: State):
        # 这里只算 hash，不调用 embedding 模型。
        ek = cache_key("query_embedding", text=state["query"], model=models.embedding_version)
        rk = cache_key(
            "retrieval",
            embedding_key=ek,
            keyword_query=state["query"],
            user=state["user"],
            filters=state["slots"],
            day=state["day"],
            index=repo.revision,
            search="bm25+cosine+rrf60+lexical-rerank-v1",
        )
        ak = cache_key(
            "answer", retrieval_key=rk, query=state["query"], generation=models.generation_version
        )
        return {
            "embedding_key": ek,
            "retrieval_key": rk,
            "answer_key": ak,
            "trace": [f"keys：构造三个 key，未计算向量；embedding key={ek[-12:]}"],
        }

    def answer_cache(state: State):
        cached = caches["answer"].get(state["answer_key"])
        return {
            "cached_answer": cached is not None,
            **(cached or {}),
            "trace": [f"answer_cache：{'HIT' if cached is not None else 'MISS'}"],
        }

    def retrieval_cache(state: State):
        cached = caches["retrieval"].get(state["retrieval_key"])
        return {
            "cached_retrieval": cached is not None,
            "docs": cached or [],
            "trace": [f"retrieval_cache：{'HIT' if cached is not None else 'MISS'}"],
        }

    def embed(state: State):
        vector = caches["embedding"].get(state["embedding_key"])
        hit = vector is not None
        if not hit:
            vector = models.embed(state["query"])
            caches["embedding"].put(state["embedding_key"], vector)
        return {
            "vector": vector,
            "trace": [f"embed：{'HIT，复用向量' if hit else 'MISS，实际编码'}"],
        }

    def retrieve(state: State):
        docs = repo.search(
            state["query"], state["vector"], state["user"], state["slots"], state["day"]
        )
        return {
            "docs": docs,
            "trace": [f"retrieve：ACL / 地区 / 日期过滤后混合检索，{len(docs)} 个候选"],
        }

    def rerank(state: State):
        # 仅作教学重排：关键词覆盖率优先，RRF 打破并列；不是 cross-encoder。
        query_tokens = set(tokens(state["query"]))
        docs = sorted(
            state["docs"],
            key=lambda d: (
                len(query_tokens & set(tokens(d["section"] + d["text"]))),
                d["rrf_score"],
            ),
            reverse=True,
        )[:3]
        caches["retrieval"].put(state["retrieval_key"], docs)
        return {"docs": docs, "trace": ["rerank：教学词项重排，缓存最终候选"]}

    def generate(state: State):
        answer = models.generate(state["query"], state["docs"])
        caches["answer"].put(state["answer_key"], {"answer": answer, "docs": state["docs"]})
        return {"answer": answer, "trace": [f"generate：{models.mode} 模式；写入答案缓存"]}

    def refuse(state: State):
        return {
            "answer": state.get("clarification") or "未找到你有权访问且当前适用的证据，请联系 HR。",
            "docs": [],
            "trace": ["refuse：澄清或证据不足，不调用生成模型"],
        }

    # node 返回的 dict 只更新 State 中相应字段。
    builder = StateGraph(State)
    for name, function in (
        ("rewrite", rewrite),
        ("keys", keys),
        ("answer_cache", answer_cache),
        ("retrieval_cache", retrieval_cache),
        ("embed", embed),
        ("retrieve", retrieve),
        ("rerank", rerank),
        ("generate", generate),
        ("refuse", refuse),
    ):
        builder.add_node(name, function)
    builder.add_edge(START, "rewrite")
    builder.add_conditional_edges(
        "rewrite", lambda s: "refuse" if s["clarification"] else "keys", ["refuse", "keys"]
    )
    builder.add_edge("keys", "answer_cache")
    builder.add_conditional_edges(
        "answer_cache",
        lambda s: END if s["cached_answer"] else "retrieval_cache",
        [END, "retrieval_cache"],
    )
    builder.add_conditional_edges(
        "retrieval_cache",
        lambda s: ("generate" if s["docs"] else "refuse") if s["cached_retrieval"] else "embed",
        ["generate", "refuse", "embed"],
    )
    builder.add_edge("embed", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_conditional_edges(
        "rerank", lambda s: "generate" if s["docs"] else "refuse", ["generate", "refuse"]
    )
    builder.add_edge("generate", END)
    builder.add_edge("refuse", END)
    return builder.compile()
