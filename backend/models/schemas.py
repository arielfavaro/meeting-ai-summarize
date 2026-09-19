from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SpeakerSegment(BaseModel):
    id: int
    start: float
    end: float
    speaker: str
    text: str


class ActionItem(BaseModel):
    task: str
    owner: str = "Não atribuído"
    deadline: str = "A definir"
    status: str = "Pendente"


class TopicItem(BaseModel):
    title: str
    discussion: str
    conclusions: Optional[str] = None


class MeetingMinutes(BaseModel):
    title: str
    date: str
    duration_minutes: float = 0.0
    participants: List[str] = Field(default_factory=list)
    executive_summary: str
    main_topics: List[TopicItem] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    open_points: List[str] = Field(default_factory=list)
    suggested_speakers: Dict[str, str] = Field(default_factory=dict)
    raw_markdown: str


class ProcessOptions(BaseModel):
    whisper_model: str = "small"
    language: str = "pt"
    ollama_model: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    custom_prompt: Optional[str] = None
    hf_token: Optional[str] = None


class MeetingDetail(BaseModel):
    id: str
    title: str
    created_at: str
    audio_filename: str
    audio_duration: float
    audio_url: str
    segments: List[SpeakerSegment] = Field(default_factory=list)
    summary: Optional[MeetingMinutes] = None
    speaker_map: Dict[str, str] = Field(default_factory=dict)


class MeetingListItem(BaseModel):
    id: str
    title: str
    created_at: str
    audio_duration: float
    speaker_count: int
    has_summary: bool


class JobStatus(BaseModel):
    job_id: str
    meeting_id: Optional[str] = None
    status: str  # "queued", "preprocessing", "diarizing", "transcribing", "summarizing", "completed", "failed"
    progress: int = 0  # 0 to 100
    current_step: str = ""
    error: Optional[str] = None
    result: Optional[MeetingDetail] = None


class UpdateSpeakersRequest(BaseModel):
    speaker_map: Dict[str, str]
    regenerate_summary: bool = False
    ollama_model: Optional[str] = None


class RegenerateSummaryRequest(BaseModel):
    ollama_model: Optional[str] = None
    custom_prompt: Optional[str] = None
