"""Cliente HTTP do Ollama (infraestrutura)."""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from backend.services.llm.base import LLMError, LLMResponse, LLMUnavailableError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 900.0, transport: Optional[httpx.AsyncBaseTransport] = None,
                 keep_alive: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._transport = transport  # injetável em testes (httpx.MockTransport)

    def _client(self, timeout: Optional[float] = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout or self.timeout, transport=self._transport)

    async def list_models(self) -> List[str]:
        try:
            async with self._client(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:  # diagnóstico: nunca deve derrubar a API
            logger.warning("Não foi possível conectar ao Ollama (%s): %s", self.base_url, e)
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
    ) -> LLMResponse:
        options: Dict[str, Any] = {"temperature": temperature, "top_p": 0.9}
        if num_ctx:
            options["num_ctx"] = num_ctx
        if num_predict:
            options["num_predict"] = num_predict

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
            # Saída estruturada: o Ollama restringe a geração ao JSON Schema (gramática).
            "format": schema if schema is not None else "json",
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        try:
            async with self._client() as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as e:
            raise LLMUnavailableError(f"Ollama inacessível em {self.base_url}: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Erro HTTP ao chamar o Ollama: {e}") from e

        if resp.status_code == 404:
            raise LLMUnavailableError(f"Modelo '{model}' não encontrado no Ollama. Baixe-o na aba Ajustes.")
        if resp.status_code >= 400:
            raise LLMError(f"Ollama retornou {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        return LLMResponse(
            content=(data.get("message") or {}).get("content", "").strip(),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            done_reason=data.get("done_reason"),
        )

    async def pull_model(self, model_name: str) -> Dict[str, str]:
        async with self._client(timeout=3600.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/pull",
                json={"model": model_name, "name": model_name, "stream": True},
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    try:
                        body = json.loads(body).get("error") or body
                    except json.JSONDecodeError:
                        pass
                    raise LLMError(f"Ollama retornou status {resp.status_code}: {body}")

                last_status = "Download em andamento..."
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in data:
                        raise LLMError(data["error"])
                    last_status = data.get("status", last_status)
                    if data.get("total"):
                        pct = data.get("completed", 0) / data["total"] * 100
                        logger.info("Ollama pull %s: %s (%.1f%%)", model_name, last_status, pct)
                return {"status": "success", "detail": last_status}
