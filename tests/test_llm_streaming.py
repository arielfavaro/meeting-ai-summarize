"""Streaming do Ollama: aborta cedo gerações degeneradas (sem esperar o limite de tokens)."""
import _env  # noqa: F401

import json
import unittest

import httpx

from backend.services.llm.loop_guard import LoopGuard
from backend.services.llm.ollama_client import OllamaClient


class TestLoopGuard(unittest.TestCase):
    def test_detects_endless_whitespace(self):
        guard = LoopGuard()
        self.assertEqual(guard.check('{"objectives": [' + "\n" * 300), "espaços em branco sem fim")

    def test_detects_repetition(self):
        guard = LoopGuard()
        text = '{"topics": [' + '{"title": "Deploy", "evidence": [1]}, ' * 20
        self.assertEqual(guard.check(text), "repetição do mesmo trecho")

    def test_normal_json_is_not_flagged(self):
        guard = LoopGuard()
        items = [{"description": f"Decisão número {i} sobre o tema {i * 7}", "evidence": [i, i + 1]} for i in range(40)]
        text = json.dumps({"decisions": items}, ensure_ascii=False, indent=2)
        for end in range(200, len(text), 150):
            self.assertIsNone(guard.check(text[:end]))


def _stream(lines):
    body = "\n".join(json.dumps(line) for line in lines).encode()
    return httpx.MockTransport(lambda request: httpx.Response(200, content=body))


class TestOllamaStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_complete_response(self):
        lines = [{"message": {"content": '{"a": '}, "done": False},
                 {"message": {"content": "1}"}, "done": False},
                 {"message": {"content": ""}, "done": True, "done_reason": "stop", "eval_count": 5,
                  "prompt_eval_count": 10}]
        client = OllamaClient("http://ollama", transport=_stream(lines))
        resp = await client.chat([{"role": "user", "content": "x"}], "m")
        self.assertEqual((resp.content, resp.done_reason, resp.completion_tokens), ('{"a": 1}', "stop", 5))

    async def test_loop_is_aborted_early(self):
        lines = [{"message": {"content": '{"objectives": ['}, "done": False}]
        lines += [{"message": {"content": "\n"}, "done": False} for _ in range(5000)]
        client = OllamaClient("http://ollama", transport=_stream(lines))
        ticks = []
        resp = await client.chat([{"role": "user", "content": "x"}], "m", on_token=ticks.append)
        self.assertEqual(resp.done_reason, "loop")
        self.assertEqual(resp.loop_reason, "espaços em branco sem fim")
        self.assertLess(resp.completion_tokens, 500)  # parou bem antes dos 5000 pedaços

    async def test_payload_uses_streaming_and_extra_options(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=json.dumps({"message": {"content": "{}"}, "done": True,
                                                           "done_reason": "stop"}).encode())

        client = OllamaClient("http://ollama", transport=httpx.MockTransport(handler), keep_alive="5m")
        await client.chat([{"role": "user", "content": "x"}], "m", num_ctx=4096, num_predict=100,
                          options={"repeat_penalty": 1.15})
        self.assertTrue(seen["stream"])
        self.assertEqual(seen["options"]["repeat_penalty"], 1.15)
        self.assertEqual(seen["options"]["num_ctx"], 4096)
        self.assertEqual(seen["keep_alive"], "5m")


if __name__ == "__main__":
    unittest.main()
