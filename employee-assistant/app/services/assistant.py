"""Service：装配依赖，维护会话，调用 graph；不处理 HTTP。"""

from datetime import date
from threading import RLock

from app.config import Settings
from app.errors import IngestForbiddenError, UnknownUserError
from app.models import Citation, HealthResponse, IngestRequest, IngestResponse, QueryResponse
from app.providers import Models
from app.repositories.policies import Cache, Repository
from app.services.graph import build_graph

USERS = {
    "alice": {"id": "alice", "groups": ["employees"], "region": "上海"},
    "helen": {"id": "helen", "groups": ["employees", "hr"], "region": "上海"},
    "bob": {"id": "bob", "groups": ["employees"], "region": "纽约"},
}


class Assistant:
    def __init__(self, path=None, models=None, settings=None):
        self.settings = settings or Settings()
        self.models = models or Models(settings=self.settings)
        self.repo = Repository(path or self.settings.policies_path, self.models)
        self.caches = {
            name: Cache(ttl=self.settings.cache_ttl)
            for name in ("answer", "retrieval", "embedding")
        }
        self.sessions = Cache(ttl=self.settings.session_ttl)
        self.graph = build_graph(self.repo, self.models, self.caches)
        # 单进程教学实现：防止并发请求交叉修改索引快照和会话。
        # 代价是串行执行，生产环境应换外部存储和更细粒度的并发控制。
        self.lock = RLock()

    def ask(self, question, user_id="alice", session_id="lesson", day=None):
        if user_id not in USERS:
            raise UnknownUserError("未知演示用户")
        if not question.strip():
            raise ValueError("问题不能为空")
        with self.lock:
            self.repo.refresh()
            session_key = (user_id, session_id)  # 相同 session_id 的不同用户不能共享历史。
            result = self.graph.invoke(
                {
                    "question": question,
                    "user": USERS[user_id],
                    "context": self.sessions.get(session_key) or {},
                    "day": day or date.today().isoformat(),
                    "trace": [],
                }
            )
            self.sessions.put(session_key, result["slots"])
            return QueryResponse(
                answer=result["answer"],
                standalone_query=result["query"],
                citations=[Citation(**d) for d in result.get("docs", [])],
                trace=result["trace"],
                mode=self.models.mode,
            )

    def ingest(self, request: IngestRequest, user_id: str) -> IngestResponse:
        if user_id not in USERS:
            raise UnknownUserError("未知演示用户")
        if "hr" not in USERS[user_id]["groups"]:
            raise IngestForbiddenError("仅 HR 演示用户可以入库")
        with self.lock:
            self.repo.refresh()
            existing = {d["id"]: d for d in self.repo.documents}
            results = []
            for policy in request.documents:
                incoming = policy.model_dump(mode="json")
                old = existing.get(policy.id)
                changed = old is None or incoming != {
                    k: v for k, v in old.items() if k != "version"
                }
                version = (old["version"] + int(changed)) if old else 1
                existing[policy.id] = {**incoming, "version": version}
                results.append({"id": policy.id, "version": version, "changed": changed})
            if any(row["changed"] for row in results):
                self.repo.replace_documents(list(existing.values()))
            return IngestResponse(
                documents=results,
                index_revision=self.repo.revision,
                chunk_count=len(self.repo.chunks),
            )

    def health(self) -> HealthResponse:
        # 进程及已加载索引状态；不执行昂贵模型请求，也不承诺外部模型就绪。
        with self.lock:
            return HealthResponse(mode=self.models.mode, **self.repo.stats())
