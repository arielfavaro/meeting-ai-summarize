"""Contratos (ports) para provedores de LLM — permite trocar Ollama por outro backend ou por fakes em testes."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol


class LLMError(Exception):
    """Falha de comunicação ou de execução no provedor de LLM."""


class LLMUnavailableError(LLMError):
    """Provedor indisponível (conexão recusada, modelo não instalado...)."""


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    done_reason: Optional[str] = None  # "stop" | "length" | "loop" (abortada por geração degenerada)
    loop_reason: Optional[str] = None


class LLMClient(Protocol):
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
    ) -> LLMResponse: ...
