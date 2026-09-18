"""完整 HTTP 入库→查询→更新→重启，以及失败原子性。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers import Models
from app.services.assistant import Assistant


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "policies.json"
        self.bot = Assistant(self.path, Models("demo"))
        self.client = self.enterContext(TestClient(create_app(assistant=self.bot)))
        self.headers = {"X-Demo-User": "helen"}
        self.document = {
            "id": "test-leave",
            "title": "测试年假政策",
            "content": "## 入口\n年假通过 Portal-A 申请。",
            "groups": ["employees"],
            "region": "上海",
            "topic": "年假",
            "effective_from": "2020-01-01",
        }

    def ingest(self, document=None):
        return self.client.post(
            "/ingest", headers=self.headers, json={"documents": [document or self.document]}
        )

    def query(self):
        return self.client.post("/query", json={"question": "年假怎么申请？"})

    def test_health_empty_and_ingest_query_update_restart(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["document_count"], 0)
        first = self.ingest()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(
            first.json()["documents"][0], {"id": "test-leave", "version": 1, "changed": True}
        )
        answer = self.query().json()
        self.assertIn("Portal-A", answer["answer"])
        self.assertIn("answer_cache：HIT", self.query().json()["trace"])
        revision = first.json()["index_revision"]
        same = self.ingest().json()
        self.assertEqual(same["index_revision"], revision)
        self.assertFalse(same["documents"][0]["changed"])
        self.document["content"] = "## 入口\n年假通过 Portal-B 申请。"
        changed = self.ingest().json()
        self.assertEqual(changed["documents"][0]["version"], 2)
        self.assertNotEqual(changed["index_revision"], revision)
        answer = self.query().json()
        self.assertIn("Portal-B", answer["answer"])
        self.assertNotIn("Portal-A", answer["answer"])
        self.assertIn("embed：HIT，复用向量", answer["trace"])
        restarted = Assistant(self.path, Models("demo"))
        self.assertIn("Portal-B", restarted.ask("年假怎么申请？").answer)

    def test_ingest_authorization(self):
        response = self.client.post("/ingest", json={"documents": [self.document]})
        self.assertEqual(response.status_code, 403)
        response = self.client.post(
            "/ingest", headers={"X-Demo-User": "nobody"}, json={"documents": [self.document]}
        )
        self.assertEqual(response.status_code, 401)
        self.assertFalse(self.path.exists())

    def test_invalid_payloads(self):
        invalid = [
            {**self.document, "version": 100},
            {**self.document, "content": "  "},
            {**self.document, "effective_to": "2019-01-01"},
            {**self.document, "groups": []},
            {**self.document, "id": "../unsafe"},
        ]
        for doc in invalid:
            with self.subTest(doc=doc):
                self.assertEqual(self.ingest(doc).status_code, 422)
        self.assertEqual(
            self.client.post("/ingest", headers=self.headers, json={"documents": []}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/ingest", headers=self.headers, json={"documents": [self.document, self.document]}
            ).status_code,
            422,
        )
        self.assertEqual(self.ingest({**self.document, "content": "## 仅标题"}).status_code, 422)
        self.assertFalse(self.path.exists())

    def test_embedding_failure_does_not_publish_partial_batch(self):
        self.ingest()
        before = self.path.read_bytes()
        revision = self.bot.repo.revision
        first = {**self.document, "id": "new-a", "content": "新的年假材料。"}
        second = {**self.document, "id": "new-b", "content": "另一份年假材料。"}
        with patch.object(
            self.bot.models, "embed", side_effect=[[0.0] * 256, httpx.ConnectError("offline")]
        ):
            response = self.client.post(
                "/ingest", headers=self.headers, json={"documents": [first, second]}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.bot.repo.revision, revision)
        self.assertEqual(self.bot.health().document_count, 1)

    def test_disk_failure_keeps_old_snapshot(self):
        self.ingest()
        before = self.path.read_bytes()
        revision = self.bot.repo.revision
        self.document["content"] = "## 入口\n年假通过 Broken-Portal 申请。"
        with patch("app.repositories.policies.os.replace", side_effect=OSError("disk failure")):
            response = self.ingest()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.bot.repo.revision, revision)
        self.assertIn("Portal-A", self.query().json()["answer"])

    def test_acl_change_invalidates_warm_answer(self):
        self.ingest()
        self.assertTrue(self.query().json()["citations"])
        self.document["groups"] = ["hr"]
        with patch.object(
            self.bot.models, "embed", side_effect=AssertionError("metadata-only change")
        ):
            self.assertEqual(self.ingest().status_code, 200)
            self.assertEqual(self.query().json()["citations"], [])

    def test_plain_text_is_indexed(self):
        self.document["content"] = "年假入口是 Plain-Portal。"
        self.assertEqual(self.ingest().status_code, 200)
        self.assertIn("Plain-Portal", self.query().json()["answer"])

    def test_corrupt_file_does_not_return_cached_answer(self):
        self.ingest()
        self.query()
        self.path.write_text("broken-json")
        self.assertEqual(self.query().status_code, 503)
