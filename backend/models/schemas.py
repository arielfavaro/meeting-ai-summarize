"""
Modelos de domínio e DTOs da API.

Convenções:
- `SpeakerSegment.speaker_id` é o identificador ESTÁVEL do locutor ("Locutor 1").
  `SpeakerSegment.speaker` é apenas o nome de exibição (desnormalizado) e é recalculado
  a partir de `MeetingDetail.speaker_map` sempre que a reunião é salva ou apresentada.
- Textos da ata (resumo, tópicos, tarefas...) referenciam locutores pelo rótulo estável
  ("Locutor 2"); os nomes reais são substituídos apenas na apresentação/exportação.
- Datas são armazenadas em ISO 8601.
"""
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Transcrição
# ---------------------------------------------------------------------------
class SpeakerSegment(BaseModel):
    id: int
    start: float
    end: float
    speaker_id: str = ""
    speaker: str
    text: str

    @model_validator(mode="after")
    def _default_speaker_id(self):
        if not self.speaker_id:
            self.speaker_id = self.speaker
        return self


# ---------------------------------------------------------------------------
# Ata de reunião
# ---------------------------------------------------------------------------
ObjectiveStatus = Literal["atingido", "parcial", "nao_atingido", "indefinido"]
ObjectiveOrigin = Literal["declarado", "inferido", "informado"]
MinutesSource = Literal["llm", "heuristic"]


class GroundedItem(BaseModel):
    """Item da ata com referência aos segmentos da transcrição que o sustentam."""
    evidence: List[int] = Field(default_factory=list)
    grounded: bool = True


class Objective(GroundedItem):
    description: str
    origin: ObjectiveOrigin = "inferido"
    status: ObjectiveStatus = "indefinido"
    notes: Optional[str] = None


class TopicItem(GroundedItem):
    title: str
    discussion: str = ""
    conclusions: Optional[str] = None


class Decision(GroundedItem):
    description: str


class OpenPoint(GroundedItem):
    description: str


class ActionItem(GroundedItem):
    task: str
    owner: str = "Não atribuído"
    deadline: str = "A definir"
    due_date: Optional[str] = None  # ISO (YYYY-MM-DD) quando o prazo pôde ser resolvido
    status: str = "Pendente"


class SpeakerSuggestion(BaseModel):
    speaker_id: str
    name: str
    confidence: float = 0.0
    kind: str = "outro"
    evidence: List[int] = Field(default_factory=list)
    applied: bool = False
    reason: Optional[str] = None


def _coerce_text_items(value, key: str = "description"):
    """Compatibilidade com atas antigas, onde decisões/pontos eram listas de strings."""
    if not isinstance(value, list):
        return value
    return [{key: v} if isinstance(v, str) else v for v in value]


class MeetingMinutes(BaseModel):
    title: str
    date: str = ""
    duration_minutes: float = 0.0
    participants: List[str] = Field(default_factory=list)
    meeting_type: Optional[str] = None
    objectives: List[Objective] = Field(default_factory=list)
    executive_summary: str = ""
    main_topics: List[TopicItem] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    open_points: List[OpenPoint] = Field(default_factory=list)
    risks: List[OpenPoint] = Field(default_factory=list)
    speaker_suggestions: List[SpeakerSuggestion] = Field(default_factory=list)

    # Metadados de geração (rastreabilidade)
    source: MinutesSource = "llm"
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    strategy: Optional[str] = None  # "single_pass" | "map_reduce" | "heuristic"
    warnings: List[str] = Field(default_factory=list)
    generated_at: Optional[str] = None

    # Renderizado na apresentação (nunca é a fonte da verdade)
    raw_markdown: str = ""

    @field_validator("decisions", "open_points", "risks", mode="before")
    @classmethod
    def _legacy_str_lists(cls, v):
        return _coerce_text_items(v)

    @model_validator(mode="before")
    @classmethod
    def _legacy_suggested_speakers(cls, data):
        # Formato antigo: {"suggested_speakers": {"Locutor 1": "Ariel"}}
        if isinstance(data, dict) and "suggested_speakers" in data and "speaker_suggestions" not in data:
            legacy = data.get("suggested_speakers") or {}
            if isinstance(legacy, dict):
                data = dict(data)
                data["speaker_suggestions"] = [
                    {"speaker_id": k, "name": v, "applied": True, "reason": "legado"}
                    for k, v in legacy.items() if isinstance(v, str) and v != k
                ]
        return data


# ---------------------------------------------------------------------------
# Contexto informado pelo usuário
# ---------------------------------------------------------------------------
MeetingType = Literal[
    "geral", "daily", "planejamento", "retrospectiva", "one_on_one",
    "comercial", "tecnica", "entrevista", "status_report"
]


class MeetingContext(BaseModel):
    objective: Optional[str] = None          # Objetivo / pauta declarada
    meeting_type: MeetingType = "geral"
    participants: List[str] = Field(default_factory=list)  # Nomes esperados
    glossary: List[str] = Field(default_factory=list)      # Termos, siglas, produtos
    custom_prompt: Optional[str] = None

    @field_validator("participants", "glossary", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.replace(";", ",").replace("\n", ",").split(",")
        return [s.strip() for s in v if s and s.strip()]


class ProcessOptions(BaseModel):
    whisper_model: str = "medium"
    language: str = "pt"
    ollama_model: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    hf_token: Optional[str] = None
    context: MeetingContext = Field(default_factory=MeetingContext)


# ---------------------------------------------------------------------------
# Reunião
# ---------------------------------------------------------------------------
SpeakerNameSource = Literal["default", "llm", "user"]


class MeetingSource(BaseModel):
    """De onde a reunião veio — permite re-diarizar/regerar reaproveitando as etapas salvas."""
    file_id: str
    tracks: List[int] = Field(default_factory=lambda: [0])
    whisper_model: str = "medium"
    language: str = "pt"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    diarization_engine: Optional[str] = None  # motor efetivamente usado


class MeetingDetail(BaseModel):
    id: str
    title: str
    created_at: str  # ISO 8601
    audio_filename: str
    audio_duration: float
    audio_url: str
    segments: List[SpeakerSegment] = Field(default_factory=list)
    summary: Optional[MeetingMinutes] = None
    speaker_map: Dict[str, str] = Field(default_factory=dict)  # speaker_id -> nome de exibição
    speaker_name_sources: Dict[str, SpeakerNameSource] = Field(default_factory=dict)
    context: MeetingContext = Field(default_factory=MeetingContext)
    source: Optional[MeetingSource] = None

    def speaker_ids(self) -> List[str]:
        seen: Dict[str, None] = {}
        for s in self.segments:
            seen.setdefault(s.speaker_id, None)
        return list(seen)

    def display_name(self, speaker_id: str) -> str:
        return self.speaker_map.get(speaker_id) or speaker_id


class MeetingListItem(BaseModel):
    id: str
    title: str
    created_at: str
    audio_duration: float
    speaker_count: int
    has_summary: bool


JobState = Literal[
    "queued", "preprocessing", "diarizing", "transcribing",
    "aligning", "summarizing", "completed", "failed"
]


LogLevel = Literal["info", "success", "warning", "error"]


class JobLogEntry(BaseModel):
    seq: int          # sequência monotônica: o frontend renderiza só o que ainda não viu
    time: str         # HH:MM:SS
    level: LogLevel = "info"
    message: str


JobKind = Literal["process", "rediarize"]


class JobStatus(BaseModel):
    job_id: str
    kind: JobKind = "process"
    meeting_id: Optional[str] = None
    status: JobState
    progress: int = 0  # 0 a 100
    current_step: str = ""
    logs: List[JobLogEntry] = Field(default_factory=list)  # últimas linhas do log em tempo real
    error: Optional[str] = None
    result: Optional[dict] = None  # visão apresentada (nomes resolvidos); não é persistida
    request: Optional[dict] = None  # parâmetros para "tentar novamente"
    created_at: float = 0.0
    updated_at: float = 0.0


class UpdateSpeakersRequest(BaseModel):
    speaker_map: Dict[str, str]  # speaker_id -> novo nome
    regenerate_summary: bool = False
    ollama_model: Optional[str] = None


class RediarizeRequest(BaseModel):
    num_speakers: Optional[int] = Field(default=None, ge=1, le=20)
    min_speakers: Optional[int] = Field(default=None, ge=1, le=20)
    max_speakers: Optional[int] = Field(default=None, ge=1, le=20)
    engine: Optional[str] = None  # "auto" | "pyannote" | "speechbrain" | "acoustic"
    ollama_model: Optional[str] = None


class MergeSpeakersRequest(BaseModel):
    source_ids: List[str] = Field(min_length=1)
    target_id: str


class ReassignSegmentsRequest(BaseModel):
    segment_ids: List[int] = Field(min_length=1)
    speaker_id: Optional[str] = None  # None = cria um novo locutor


class RegenerateSummaryRequest(BaseModel):
    ollama_model: Optional[str] = None
    custom_prompt: Optional[str] = None
