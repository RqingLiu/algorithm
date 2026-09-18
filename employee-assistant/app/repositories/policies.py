"""Data：教学内存索引、缓存与文档更新；生产环境再换数据库。"""

import copy
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.errors import StorageError
from app.models import StoredPolicy
from app.providers import tokens


def cache_key(namespace: str, **inputs) -> str:
    serialized = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return namespace + ":" + hashlib.sha256(serialized.encode()).hexdigest()


class Cache:
    """精确缓存；值含 TTL。复制读写避免其他请求意外修改缓存内容。"""

    def __init__(self, ttl=300, capacity=1024):
        self.ttl, self.capacity = ttl, capacity
        self.values = {}

    def get(self, key):
        entry = self.values.get(key)
        if entry is None:
            return None
        expires, value = entry
        if expires <= time.monotonic():
            del self.values[key]
            return None
        return copy.deepcopy(value)

    def put(self, key, value):
        if key not in self.values and len(self.values) >= self.capacity:
            del self.values[next(iter(self.values))]
        self.values[key] = (time.monotonic() + self.ttl, copy.deepcopy(value))


def split_sections(document: dict) -> list[dict]:
    """输入已解析 Markdown。按二级标题分块；长段按中文句号，再按字符兜底。

    400 是字符上限，不是 token 上限；这里没有 PDF/OCR 解析器。
    """
    result = []
    content = document["content"]
    headings = list(re.finditer(r"(?m)^## ([^\n]+)", content))
    sections = []
    prefix = content[: headings[0].start()] if headings else content
    if prefix.strip():
        sections.append((document["title"], prefix))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
        sections.append((heading.group(1), content[heading.end() : end]))
    for section, body in sections:
        pieces, current = [], ""
        for sentence in re.findall(r"[^。]+。?", body.strip()):
            for start in range(0, len(sentence), 400):
                part = sentence[start : start + 400]
                if len(current) + len(part) > 400:
                    pieces.append(current)
                    current = ""
                current += part
        if current:
            pieces.append(current)
        for piece in pieces:
            metadata = {k: v for k, v in document.items() if k != "content"}
            result.append(
                {
                    **metadata,
                    "chunk_id": f"{document['id']}:v{document['version']}:{len(result)}",
                    "section": section,
                    "text": piece,
                }
            )
    return result


class Repository:
    def __init__(self, path: Path, models):
        self.path, self.models = path, models
        self.document_embeddings = Cache(ttl=86400)
        self.revision = ""
        self.chunks = []
        self.documents = []
        self.refresh()

    def refresh(self):
        # 每次请求检查小 JSON 文件的内容 hash，仅为演示增量发布。
        # 构建成功才替换快照；异常则请求失败，不继续回答旧快照。
        documents = self.read_documents()
        chunks, revision = self.build_snapshot(documents)
        self.documents, self.chunks, self.revision = documents, chunks, revision

    def read_documents(self):
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("知识库必须是文档列表")
            documents = [StoredPolicy.model_validate(d).model_dump(mode="json") for d in raw]
            if len({d["id"] for d in documents}) != len(documents):
                raise ValueError("知识库存在重复文档 id")
            return documents
        except (OSError, ValueError) as error:
            raise StorageError("无法读取或校验知识库文件") from error

    def build_snapshot(self, documents):
        revision = cache_key(
            "index",
            documents=documents,
            parser="sections-v1",
            embedding=self.models.embedding_version,
        )
        if revision == self.revision:
            return self.chunks, revision
        chunks = []
        for document in documents:
            document_chunks = split_sections(document)
            if not document_chunks:
                raise ValueError("文档必须包含可索引的正文，不能只有标题")
            for chunk in document_chunks:
                encoded = chunk["title"] + "\n" + chunk["section"] + "\n" + chunk["text"]
                key = cache_key("doc_embedding", text=encoded, model=self.models.embedding_version)
                vector = self.document_embeddings.get(key)
                if vector is None:
                    vector = self.models.embed(encoded)
                    self.document_embeddings.put(key, vector)
                chunks.append({**chunk, "vector": vector, "tokens": tokens(encoded)})
        return chunks, revision

    def replace_documents(self, documents):
        """先构建完整快照，再原子替换文件；单进程，由 Service 锁保护。"""
        chunks, revision = self.build_snapshot(documents)
        temporary_path = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".policies-",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                json.dump(documents, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as error:
            raise StorageError("知识库保存失败，原数据未发布更新") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        self.documents, self.chunks, self.revision = documents, chunks, revision

    def stats(self):
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "index_revision": self.revision,
        }

    def eligible(self, user, slots, day):
        return [
            d
            for d in self.chunks
            if set(d["groups"]) & set(user["groups"])
            and d["region"] in {"all", user["region"]}
            and d["status"] == "published"
            and d["effective_from"] <= day
            and (d["effective_to"] is None or day < d["effective_to"])
            and d["topic"] == slots.get("topic")
            and (not slots.get("destination") or d["destination"] == slots["destination"])
        ]

    def search(self, query, vector, user, slots, day):
        # 两路搜索都只看已授权、适用且有效的候选。
        candidates = self.eligible(user, slots, day)
        if not candidates:
            return []
        bm25 = BM25Okapi([d["tokens"] for d in candidates])
        sparse = bm25.get_scores(tokens(query))
        dense = [sum(a * b for a, b in zip(vector, d["vector"])) for d in candidates]
        rankings = [
            sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:6]
            for scores in (sparse, dense)
        ]
        fusion = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking, 1):
                fusion[i] = fusion.get(i, 0) + 1 / (60 + rank)
        return [
            {k: v for k, v in candidates[i].items() if k not in {"vector", "tokens"}}
            | {"rrf_score": fusion[i]}
            for i in sorted(fusion, key=fusion.get, reverse=True)
        ]
