"""Model：接口数据结构，以及 LangGraph 节点之间传递的 State。"""

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PolicyInput(BaseModel):
    """客户端不能指定 version；入库时按 id 自动递增。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=100_000)
    groups: list[Literal["employees", "hr"]] = Field(min_length=1)
    region: str = Field(min_length=1, max_length=100)
    topic: Literal["住宿", "年假", "薪酬"]
    destination: str | None = Field(default=None, max_length=100)
    status: Literal["published", "draft", "inactive"] = "published"
    effective_from: date = Field(default_factory=date.today)
    effective_to: date | None = None

    @model_validator(mode="after")
    def check_dates(self):
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to 必须晚于 effective_from")
        self.groups = sorted(set(self.groups))
        return self


class StoredPolicy(PolicyInput):
    version: int = Field(ge=1)


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[PolicyInput] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [d.id for d in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("同一批次不能包含重复文档 id")
        return self


class IngestedDocument(BaseModel):
    id: str
    version: int
    changed: bool


class IngestResponse(BaseModel):
    documents: list[IngestedDocument]
    index_revision: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    mode: str
    document_count: int
    chunk_count: int
    index_revision: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="lesson", min_length=1, max_length=100)


class Citation(BaseModel):
    chunk_id: str
    title: str
    section: str
    version: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    standalone_query: str
    citations: list[Citation]
    trace: list[str]
    mode: str


class State(TypedDict, total=False):
    question: str
    user: dict
    context: dict
    slots: dict
    query: str
    clarification: str
    day: str
    embedding_key: str
    retrieval_key: str
    answer_key: str
    cached_answer: bool
    cached_retrieval: bool
    vector: list[float]
    docs: list[dict]
    answer: str
    # reducer：节点返回的新 trace 会追加，而不是覆盖旧列表。
    trace: Annotated[list[str], operator.add]
