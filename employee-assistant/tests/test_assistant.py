"""运行 python -m unittest -v；验证权限、缓存与会话这些真实行为。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers import Models
from app.repositories.policies import Cache
from app.services.assistant import Assistant


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "policies.json"
        self.path.write_text((Path(__file__).parent / "fixtures" / "policies.json").read_text())
        self.bot = Assistant(self.path, Models("demo"))

    def ask(self, q, user="alice", session="test", day="2026-09-16"):
        return self.bot.ask(q, user, session, day)

    def test_acl_even_after_hr_warmed_cache(self):
        hr = self.ask("薪酬审阅流程是什么？", "helen")
        employee = self.ask("薪酬审阅流程是什么？")
        self.assertTrue(hr.citations)
        self.assertEqual(employee.citations, [])
        self.assertNotIn("联合审批", employee.answer)

    def test_region_and_embedding_reuse(self):
        self.ask("年假怎么申请？")
        other = self.ask("年假怎么申请？", "bob")
        self.assertIn("NY-HR", other.answer)
        self.assertIn("embed：HIT，复用向量", other.trace)
        self.assertIn("answer_cache：MISS", other.trace)

    def test_expired_and_future_excluded(self):
        result = self.ask("北京住宿上限是多少？")
        self.assertIn("800", result.answer)
        self.assertNotIn("999", result.answer)
        self.assertNotIn("2000", result.answer)

    def test_multiturn_and_session_isolation(self):
        self.ask("去北京出差住宿需要什么材料？")
        result = self.ask("那深圳呢？")
        self.assertIn("深圳", result.standalone_query)
        self.assertIn("材料", result.standalone_query)
        self.assertNotIn("北京", result.standalone_query)
        self.assertEqual(self.ask("那深圳呢？", "helen").citations, [])
        self.assertEqual(self.ask("那深圳呢？", session="new").citations, [])

    def test_answer_and_retrieval_cache_short_circuit(self):
        self.ask("年假怎么申请？")
        with patch.object(self.bot.models, "embed", side_effect=AssertionError("不应编码")):
            hit = self.ask("年假怎么申请？")
            self.assertIn("answer_cache：HIT", hit.trace)
            self.bot.caches["answer"].values.clear()
            hit = self.ask("年假怎么申请？")
            self.assertIn("retrieval_cache：HIT", hit.trace)
            self.assertFalse(any(t.startswith("retrieve：") for t in hit.trace))

    def test_document_update_invalidates_answers(self):
        self.ask("北京住宿上限是多少？")
        documents = json.loads(self.path.read_text())
        documents[0]["content"] = documents[0]["content"].replace("800", "850")
        documents[0]["version"] = 3
        self.path.write_text(json.dumps(documents))
        result = self.ask("北京住宿上限是多少？")
        self.assertIn("850", result.answer)
        self.assertNotIn("800", result.answer)
        self.assertIn("answer_cache：MISS", result.trace)
        self.assertIn("embed：HIT，复用向量", result.trace)

    def test_acl_update_reuses_document_vectors_but_revokes_access(self):
        self.ask("年假怎么申请？")
        documents = json.loads(self.path.read_text())
        documents[2]["groups"] = ["hr"]
        self.path.write_text(json.dumps(documents))
        with patch.object(
            self.bot.models, "embed", side_effect=AssertionError("仅 ACL 改变不应重算向量")
        ):
            result = self.ask("年假怎么申请？")
        self.assertEqual(result.citations, [])

    def test_deletion_invalidates_cache(self):
        self.ask("年假怎么申请？")
        documents = json.loads(self.path.read_text())
        self.path.write_text(json.dumps([d for d in documents if d["id"] != "leave"]))
        self.assertEqual(self.ask("年假怎么申请？").citations, [])

    def test_unknown_and_personal_balance(self):
        self.assertEqual(self.ask("明天天气如何？").citations, [])
        self.assertIn("HR 系统", self.ask("我年假还剩几天？").answer)

    def test_expiration_without_file_change(self):
        before = self.ask("北京住宿上限是多少？", day="2025-12-31")
        after = self.ask("北京住宿上限是多少？", day="2026-01-01")
        self.assertIn("999", before.answer)
        self.assertNotIn("999", after.answer)

    def test_cache_ttl(self):
        cache = Cache(ttl=1)
        with patch("app.repositories.policies.time.monotonic", return_value=10):
            cache.put("key", [1])
            self.assertEqual(cache.get("key"), [1])
        with patch("app.repositories.policies.time.monotonic", return_value=12):
            self.assertIsNone(cache.get("key"))

    def test_http(self):
        with TestClient(create_app(assistant=self.bot)) as client:
            self.assertEqual(client.get("/docs").status_code, 200)
            self.assertEqual(
                client.post("/query", json={"question": "年假怎么申请？"}).status_code, 200
            )
            self.assertEqual(client.post("/query", json={"question": " "}).status_code, 422)
            self.assertEqual(client.post("/query", json={}).status_code, 422)
            self.assertEqual(
                client.post(
                    "/query", headers={"X-Demo-User": "unknown"}, json={"question": "年假"}
                ).status_code,
                401,
            )


if __name__ == "__main__":
    unittest.main()
