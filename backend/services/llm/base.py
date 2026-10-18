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
    # Telemetria local do provedor (segundos): carga do modelo, leitura do prompt e geração.
    load_seconds: Optional[float] = None
    prompt_seconds: Optional[float] = None
    eval_seconds: Optional[float] = None

    def timing_summary(self) -> Optional[str]:
        """Ex.: "carga 41.2s · prompt 11583 tok em 6.1s (1899 tok/s) · geração 812 tok a 38 tok/s"."""
        parts = []
        if self.load_seconds is not None and self.load_seconds >= 0.5:
            parts.append(f"carga do modelo {self.load_seconds:.1f}s")
        if self.prompt_seconds and self.prompt_tokens:
            parts.append(f"prompt {self.prompt_tokens} tok em {self.prompt_seconds:.1f}s "
                         f"({self.prompt_tokens / self.prompt_seconds:.0f} tok/s)")
        if self.eval_seconds and self.completion_tokens:
            parts.append(f"geração {self.completion_tokens} tok a {self.completion_tokens / self.eval_seconds:.0f} tok/s")
        return " · ".join(parts) or None


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
