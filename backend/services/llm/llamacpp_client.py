"""
Cliente do `llama-server` (llama.cpp) via API compatível com OpenAI — infraestrutura local.

Por que existe além do Ollama: modelos MoE grandes (ex.: Qwen3.6-35B-A3B) só cabem numa GPU
de 16 GB deixando os "experts" na RAM (`--n-cpu-moe`), controle que o llama-server dá
diretamente. A interface é a mesma do OllamaClient (`chat`, `list_models`, `pull_model`).

- Saída estruturada: `response_format.json_schema` (o llama-server converte em gramática).
- Streaming com o mesmo LoopGuard do Ollama (aborta laços cedo).
- `num_ctx` é fixado na subida do servidor (`-c`); aqui só é usado para o planejamento.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

from backend.services.llm.base import LLMError, LLMResponse, LLMUnavailableError
from backend.services.llm.loop_guard import LoopGuard

logger = logging.getLogger(__name__)


class LlamaCppClient:
    def __init__(self, base_url: str, timeout: float = 900.0, transport: Optional[httpx.AsyncBaseTransport] = None,
                 disable_thinking: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self._transport = transport

    def _client(self, timeout: Optional[float] = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout or self.timeout, transport=self._transport)

    async def list_models(self) -> List[str]:
        try:
            async with self._client(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]
        except Exception as e:  # diagnóstico: nunca deve derrubar a API
            logger.warning("Não foi possível conectar ao llama-server (%s): %s", self.base_url, e)
            return []

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        temperature: float = 0.1,
        options: Optional[Dict[str, Any]] = None,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": 0.9,
        }
        if num_predict:
            payload["max_tokens"] = num_predict
        if schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "ata", "schema": schema, "strict": True}}
        else:
            payload["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            # Qwen3.x e similares: sem bloco de "pensamento" antes do JSON (a gramática já guia a saída).
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.update(options or {})  # repeat_penalty / repeat_last_n são aceitos pelo llama-server

        guard = LoopGuard()
        content = ""
        chunks = 0
        result = LLMResponse(content="")
        try:
            async with self._client() as client:
                async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise LLMError(f"llama-server retornou {resp.status_code}: {body[:500]}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if "error" in data:
                            raise LLMError(f"llama-server: {data['error']}")
                        choice = (data.get("choices") or [{}])[0]
                        content += (choice.get("delta") or {}).get("content") or ""
                        chunks += 1
                        if on_token and chunks % 50 == 0:
                            on_token(chunks)
                        self._read_usage(data, result)
                        if choice.get("finish_reason"):
                            result.done_reason = "length" if choice["finish_reason"] == "length" else "stop"
                            continue  # o último pedaço pode trazer as métricas
                        reason = guard.check(content)
                        if reason:
                            result.done_reason, result.loop_reason = "loop", reason
                            result.completion_tokens = chunks
                            break
        except httpx.ConnectError as e:
            raise LLMUnavailableError(f"llama-server inacessível em {self.base_url}: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Erro HTTP ao chamar o llama-server: {e}") from e

        result.content = content.strip()
        result.done_reason = result.done_reason or "stop"
        if result.completion_tokens is None:
            result.completion_tokens = chunks
        return result

    @staticmethod
    def _read_usage(data: Dict[str, Any], result: LLMResponse) -> None:
        usage = data.get("usage") or {}
        if usage:
            result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens)
            result.completion_tokens = usage.get("completion_tokens", result.completion_tokens)
        timings = data.get("timings") or {}
        if timings:
            result.prompt_tokens = timings.get("prompt_n", result.prompt_tokens)
            result.completion_tokens = timings.get("predicted_n", result.completion_tokens)
            if timings.get("prompt_ms") is not None:
                result.prompt_seconds = timings["prompt_ms"] / 1000.0
            if timings.get("predicted_ms") is not None:
                result.eval_seconds = timings["predicted_ms"] / 1000.0

    async def pull_model(self, model_name: str) -> Dict[str, str]:
        raise LLMError("O llama-server não baixa modelos pela interface: coloque o arquivo .gguf em "
                       "data/llamacpp e ajuste LLAMACPP_MODEL_FILE no .env.")
