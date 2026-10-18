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
        subjects = ["deploy", "orçamento", "contratação", "rollback", "auditoria", "treinamento", "fornecedor",
                    "cronograma", "migração", "backup", "segurança", "licenças", "contrato", "infraestrutura"]
        verbs = ["Aprovar", "Revisar", "Adiar", "Cancelar", "Priorizar", "Documentar", "Validar"]
        items = [{"description": f"{verbs[(i // len(subjects)) % len(verbs)]} {subjects[i % len(subjects)]} da fase {i}",
                  "evidence": [i, i + 1, i + 7]} for i in range(40)]
        topics = [{"title": f"Tópico {s}", "discussion": f"Discussão sobre {s}.", "conclusions": "Sem conclusão",
                   "evidence": [3]} for s in subjects]
        text = json.dumps({"topics": topics, "decisions": items}, ensure_ascii=False, indent=2)
        for end in range(200, len(text), 150):
            self.assertIsNone(guard.check(text[:end]))

    def test_detects_near_duplicate_items_that_only_change_numbers(self):
        guard = LoopGuard()
        items = "".join(f'{{"description": "Revisar o contrato do fornecedor de limpeza", "evidence": [{i}, {i + 1}]}}, '
                        for i in range(10, 30))
        self.assertIsNotNone(guard.check('{"decisions": [' + items))

    def test_detects_same_item_repeated_with_other_items_between(self):
        guard = LoopGuard()
        text = '{"action_items": ['
        for i in range(4):
            text += ('{"task": "Enviar o relatório de custos para a diretoria", "owner": "Locutor 1", '
                     f'"deadline": "A definir", "evidence": [{i}]}}, '
                     f'{{"task": "Tarefa diferente número {"abcd"[i]} com outro texto", "owner": "Locutor 2", '
                     '"deadline": "sexta", "evidence": [2]}, ')
        self.assertEqual(guard.check(text), "itens repetidos")

    def test_detects_endless_number_list(self):
        guard = LoopGuard()
        text = '{"decisions": [{"description": "Aprovar deploy", "evidence": [' + ", ".join(str(i) for i in range(1, 400))
        self.assertEqual(guard.check(text), "repetição do mesmo trecho")


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


def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


class TestLlamaCppStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_structured_request_and_metrics(self):
        from backend.services.llm.llamacpp_client import LlamaCppClient
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            chunks = [{"choices": [{"delta": {"content": '{"a": '}}]},
                      {"choices": [{"delta": {"content": "1}"}, "finish_reason": "stop"}],
                       "timings": {"prompt_n": 1000, "prompt_ms": 500.0, "predicted_n": 5, "predicted_ms": 100.0}}]
            return httpx.Response(200, content=_sse(chunks))

        client = LlamaCppClient("http://llama", transport=httpx.MockTransport(handler))
        resp = await client.chat([{"role": "user", "content": "x"}], "qwen", schema={"type": "object"},
                                 num_predict=300, options={"repeat_penalty": 1.15})
        self.assertEqual((resp.content, resp.done_reason), ('{"a": 1}', "stop"))
        self.assertEqual(seen["response_format"]["json_schema"]["schema"], {"type": "object"})
        self.assertEqual((seen["max_tokens"], seen["repeat_penalty"]), (300, 1.15))
        self.assertFalse(seen["chat_template_kwargs"]["enable_thinking"])
        self.assertIn("prompt 1000 tok em 0.5s (2000 tok/s)", resp.timing_summary())

    async def test_length_and_loop(self):
        from backend.services.llm.llamacpp_client import LlamaCppClient
        cut = [{"choices": [{"delta": {"content": '{"a": "xx'}, "finish_reason": "length"}]}]
        client = LlamaCppClient("http://llama", transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=_sse(cut))))
        self.assertEqual((await client.chat([], "m")).done_reason, "length")

        loop = [{"choices": [{"delta": {"content": '{"objectives": ['}}]}]
        loop += [{"choices": [{"delta": {"content": "\n"}}]} for _ in range(3000)]
        client = LlamaCppClient("http://llama", transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=_sse(loop))))
        resp = await client.chat([], "m")
        self.assertEqual(resp.done_reason, "loop")


if __name__ == "__main__":
    unittest.main()
