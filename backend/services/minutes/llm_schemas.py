"""
Esquemas de SAÍDA do LLM. São enviados ao Ollama como JSON Schema (`format`), o que
restringe a geração por gramática, e validados com Pydantic na volta.

Mantidos separados do domínio (`backend.models.schemas`) de propósito: o contrato com o
LLM é enxuto e pode evoluir (versão do prompt) sem afetar o que é persistido.
"""
from typing import List, Literal

from pydantic import BaseModel, Field

Evidence = Field(description="IDs numéricos (#) dos trechos da transcrição que sustentam o item")


class LLMObjective(BaseModel):
    description: str
    origin: Literal["declarado", "inferido", "informado"]
    status: Literal["atingido", "parcial", "nao_atingido", "indefinido"]
    notes: str = Field(description="Justificativa curta do status")
    evidence: List[int] = Evidence


class LLMTopic(BaseModel):
    title: str
    discussion: str
    conclusions: str
    evidence: List[int] = Evidence


class LLMDecision(BaseModel):
    description: str
    evidence: List[int] = Evidence


class LLMAction(BaseModel):
    task: str
    owner: str = Field(description='Rótulo do responsável (ex.: "Locutor 2"), nome citado ou "Não atribuído"')
    deadline: str = Field(description='Prazo como foi dito (ex.: "sexta-feira") ou "A definir"')
    evidence: List[int] = Evidence


class LLMItem(BaseModel):
    description: str
    evidence: List[int] = Evidence


class LLMSpeakerName(BaseModel):
    speaker_id: str = Field(description='Rótulo exato, ex.: "Locutor 3"')
    name: str
    kind: Literal["auto_apresentacao", "chamado_pelo_nome", "outro"]
    confidence: float = Field(description="0 a 1")
    evidence: List[int] = Evidence


class LLMExtraction(BaseModel):
    """Saída da etapa MAP (um bloco da reunião)."""
    objectives: List[LLMObjective]
    topics: List[LLMTopic]
    decisions: List[LLMDecision]
    action_items: List[LLMAction]
    open_points: List[LLMItem]
    risks: List[LLMItem]
    speaker_names: List[LLMSpeakerName]


class LLMMinutes(LLMExtraction):
    """Saída final (single pass ou REDUCE)."""
    title: str
    executive_summary: str
