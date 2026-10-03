"""
Esquemas de SAÍDA do LLM. São enviados ao Ollama como JSON Schema (`format`), o que
restringe a geração por gramática, e validados com Pydantic na volta.

Os limites de tamanho (maxItems / maxLength) entram SÓ no JSON Schema enviado ao modelo
(`json_schema_extra`), não na validação. Eles servem para:
- dimensionar `num_predict`/`num_ctx` ANTES da chamada (pior caso conhecido), evitando
  gerar, estourar o limite de tokens e ter que refazer tudo;
- impedir laços degenerados (o modelo repetindo itens até o fim da janela).
Se o backend não aplicar os limites, o pós-processamento corta as listas.

Mantidos separados do domínio (`backend.models.schemas`) de propósito: o contrato com o
LLM é enxuto e pode evoluir (versão do prompt) sem afetar o que é persistido.
"""
import copy
from typing import Any, Dict, List, Literal, Optional, Sequence, Type

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Limites (fonte única: usados no schema, no prompt e no orçamento de tokens)
# ---------------------------------------------------------------------------
LIMITS: Dict[str, int] = {
    "title": 120,
    "executive_summary": 1200,
    "objectives": 5, "objective_text": 200, "objective_notes": 200,
    "topics": 8, "topic_title": 100, "topic_discussion": 450, "topic_conclusions": 220,
    "decisions": 12, "decision_text": 240,
    "action_items": 15, "action_task": 200, "action_owner": 60, "action_deadline": 60,
    "open_points": 10, "risks": 8, "item_text": 200,
    "evidence": 5,
    "speaker_name": 60, "speaker_evidence": 3,
}


def _text(limit_key: str, description: Optional[str] = None) -> Any:
    return Field(description=description, json_schema_extra={"maxLength": LIMITS[limit_key]})


def _items(limit_key: str, description: Optional[str] = None) -> Any:
    return Field(description=description, json_schema_extra={"maxItems": LIMITS[limit_key]})


def _evidence(limit_key: str = "evidence") -> Any:
    return Field(description="IDs numéricos (#) dos trechos da transcrição que sustentam o item",
                 json_schema_extra={"maxItems": LIMITS[limit_key]})


class LLMObjective(BaseModel):
    description: str = _text("objective_text")
    origin: Literal["declarado", "inferido", "informado"]
    status: Literal["atingido", "parcial", "nao_atingido", "indefinido"]
    notes: str = _text("objective_notes", "Justificativa curta do status")
    evidence: List[int] = _evidence()


class LLMTopic(BaseModel):
    title: str = _text("topic_title")
    discussion: str = _text("topic_discussion")
    conclusions: str = _text("topic_conclusions")
    evidence: List[int] = _evidence()


class LLMDecision(BaseModel):
    description: str = _text("decision_text")
    evidence: List[int] = _evidence()


class LLMAction(BaseModel):
    task: str = _text("action_task")
    owner: str = _text("action_owner", 'Rótulo do responsável (ex.: "Locutor 2"), nome citado ou "Não atribuído"')
    deadline: str = _text("action_deadline", 'Prazo como foi dito (ex.: "sexta-feira") ou "A definir"')
    evidence: List[int] = _evidence()


class LLMItem(BaseModel):
    description: str = _text("item_text")
    evidence: List[int] = _evidence()


class LLMSpeakerName(BaseModel):
    speaker_id: str = Field(description='Rótulo exato, ex.: "Locutor 3"')
    name: str = _text("speaker_name")
    kind: Literal["auto_apresentacao", "chamado_pelo_nome", "outro"]
    confidence: float = Field(description="0 a 1")
    evidence: List[int] = _evidence("speaker_evidence")


class LLMExtraction(BaseModel):
    """Saída da etapa MAP (um bloco da reunião)."""
    objectives: List[LLMObjective] = _items("objectives")
    topics: List[LLMTopic] = _items("topics")
    decisions: List[LLMDecision] = _items("decisions")
    action_items: List[LLMAction] = _items("action_items")
    open_points: List[LLMItem] = _items("open_points")
    risks: List[LLMItem] = _items("risks")
    speaker_names: List[LLMSpeakerName]


class LLMMinutes(LLMExtraction):
    """Saída final (single pass ou REDUCE)."""
    title: str = _text("title")
    executive_summary: str = _text("executive_summary")


# ---------------------------------------------------------------------------
# Schema por requisição + orçamento de saída
# ---------------------------------------------------------------------------
# Um bloco do map-reduce cobre ~15-20 min: listas menores que as da ata final.
EXTRACT_LIST_LIMITS: Dict[str, int] = {
    "objectives": 3, "topics": 4, "decisions": 6, "action_items": 8, "open_points": 5, "risks": 4,
}
FINAL_LIST_LIMITS: Dict[str, int] = {k: LIMITS[k] for k in EXTRACT_LIST_LIMITS}


def schema_for(model: Type[BaseModel], speaker_ids: Sequence[str],
               list_limits: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    JSON Schema enviado ao Ollama, ajustado à reunião: `speaker_id` vira enum dos rótulos
    reais (o modelo não consegue inventar "Locutor 9"), a lista de nomes é limitada ao
    número de locutores e as listas principais podem ter limites próprios (ex.: por bloco).
    """
    schema = copy.deepcopy(model.model_json_schema())
    props = schema.get("properties", {})
    defs = schema.get("$defs", {})
    if "LLMSpeakerName" in defs and speaker_ids:
        defs["LLMSpeakerName"]["properties"]["speaker_id"]["enum"] = list(speaker_ids)
    if "speaker_names" in props:
        props["speaker_names"]["maxItems"] = max(1, len(speaker_ids))
    for key, limit in (list_limits or {}).items():
        if key in props:
            props[key]["maxItems"] = limit
    return schema


# Caracteres por token na SAÍDA (JSON + português). Conservador de propósito: superestimar
# só reserva memória; subestimar faz a resposta ser cortada e refeita.
OUTPUT_CHARS_PER_TOKEN = 2.8
_INT_CHARS = 6          # "1234, "
_NUMBER_CHARS = 6       # 0.95
_UNBOUNDED_STRING = 120


def _max_chars(node: Dict[str, Any], defs: Dict[str, Any]) -> int:
    if "$ref" in node:
        return _max_chars(defs[node["$ref"].split("/")[-1]], defs)
    if "enum" in node:
        return max(len(str(v)) for v in node["enum"]) + 2
    t = node.get("type")
    if t == "object":
        props = node.get("properties", {})
        return 2 + sum(len(k) + 4 + _max_chars(v, defs) for k, v in props.items())
    if t == "array":
        return 2 + node.get("maxItems", 10) * (_max_chars(node.get("items", {}), defs) + 2)
    if t == "string":
        return node.get("maxLength", _UNBOUNDED_STRING) + 2
    if t == "integer":
        return _INT_CHARS
    if t == "number":
        return _NUMBER_CHARS
    return 8


def max_output_tokens(schema: Dict[str, Any]) -> int:
    """Pior caso de tokens que uma resposta válida para o schema pode ter."""
    chars = _max_chars(schema, schema.get("$defs", {}))
    return int(chars / OUTPUT_CHARS_PER_TOKEN) + 64


def size_rules_text(list_limits: Optional[Dict[str, int]] = None) -> str:
    """Limites em linguagem natural para o prompt (o modelo planeja o texto e não é cortado no meio)."""
    L = {**LIMITS, **(list_limits or {})}
    return (
        "LIMITES DE TAMANHO (seja conciso e priorize o que é relevante):\n"
        f"- até {L['objectives']} objetivos, {L['topics']} tópicos, {L['decisions']} decisões, "
        f"{L['action_items']} tarefas, {L['open_points']} pontos em aberto e {L['risks']} riscos;\n"
        f"- até {L['evidence']} evidências por item (as mais representativas);\n"
        f"- discussão de cada tópico com até {L['topic_discussion']} caracteres; demais textos curtos "
        f"(até ~{L['item_text']} caracteres); resumo executivo com até {L['executive_summary']} caracteres."
    )
