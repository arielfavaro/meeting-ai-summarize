"""Dublês de teste (sem modelos de IA)."""
import json
from typing import Any, Dict, List, Optional

from backend.models.schemas import SpeakerSegment
from backend.services.llm.base import LLMResponse, LLMUnavailableError


class FakeLLM:
    """Devolve respostas roteirizadas em ordem e registra as chamadas."""

    def __init__(self, responses: Optional[List[Any]] = None, unavailable: bool = False):
        self.responses = list(responses or [])
        self.unavailable = unavailable
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, messages, model, *, schema=None, num_ctx=None, num_predict=None, temperature=0.1,
                   options=None, on_token=None):
        self.calls.append({"messages": messages, "model": model, "schema": schema, "num_ctx": num_ctx,
                           "num_predict": num_predict, "temperature": temperature, "options": options})
        if self.unavailable:
            raise LLMUnavailableError("Ollama offline (fake)")
        item = self.responses.pop(0)
        if isinstance(item, LLMResponse):
            return item
        content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return LLMResponse(content=content, prompt_tokens=100, completion_tokens=50, done_reason="stop")


def seg(i, start, speaker, text):
    return SpeakerSegment(id=i, start=start, end=start + 3.0, speaker=speaker, text=text)


def sample_segments() -> List[SpeakerSegment]:
    return [
        seg(1, 0.0, "Locutor 1", "Bom dia! Aqui é o Carlos. A ideia hoje é decidir a data do deploy."),
        seg(2, 4.0, "Locutor 2", "Perfeito. Mariana, você consegue validar os testes de carga?"),
        seg(3, 8.0, "Locutor 3", "Consigo sim, entrego os resultados até sexta-feira."),
        seg(4, 12.0, "Locutor 1", "Então fica decidido: deploy em produção na segunda que vem."),
        seg(5, 16.0, "Locutor 2", "Ainda falta definir quem cuida do rollback."),
    ]


def minutes_payload(**overrides) -> Dict[str, Any]:
    data = {
        "objectives": [{"description": "Decidir a data do deploy", "origin": "declarado", "status": "atingido",
                        "notes": "Data definida para segunda.", "evidence": [1, 4]}],
        "topics": [{"title": "Deploy em produção", "discussion": "Locutor 1 propôs a data.",
                    "conclusions": "Deploy na segunda.", "evidence": [1, 4]}],
        "decisions": [{"description": "Deploy em produção na segunda-feira", "evidence": [4]}],
        "action_items": [
            {"task": "Validar os testes de carga", "owner": "Locutor 3", "deadline": "sexta-feira", "evidence": [2, 3]},
            {"task": "Validar testes de carga", "owner": "Locutor 3", "deadline": "sexta-feira", "evidence": [3]},
            {"task": "Preparar plano de comunicação", "owner": "", "deadline": "A definir", "evidence": [999]},
        ],
        "open_points": [{"description": "Responsável pelo rollback", "evidence": [5]}],
        "risks": [],
        "speaker_names": [
            {"speaker_id": "Locutor 1", "name": "Carlos", "kind": "auto_apresentacao", "confidence": 0.95, "evidence": [1]},
            {"speaker_id": "Locutor 3", "name": "Mariana", "kind": "chamado_pelo_nome", "confidence": 0.9, "evidence": [2]},
            {"speaker_id": "Locutor 2", "name": "Mariana", "kind": "chamado_pelo_nome", "confidence": 0.6, "evidence": [2]},
        ],
        "title": "Definição da data de deploy",
        "executive_summary": "Locutor 1 conduziu a reunião e Locutor 3 assumiu os testes.",
    }
    data.update(overrides)
    return data
