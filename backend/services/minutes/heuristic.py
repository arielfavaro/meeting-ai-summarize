"""Gerador de contingência (sem LLM). Sempre marca a ata como `source="heuristic"`."""
import re
from datetime import datetime
from typing import List, Optional, Sequence

from backend.models.schemas import (
    ActionItem, Decision, MeetingContext, MeetingMinutes, Objective, SpeakerSegment, TopicItem,
)

ACTION_PATTERN = re.compile(
    r"\b(vou|vamos|preciso|precisamos|fico respons[aá]vel|me comprometo|tarefa|enviar|entregar|"
    r"verificar|alinhar|criar|desenvolver|agendar|revisar)\b", re.IGNORECASE)
DECISION_PATTERN = re.compile(
    r"\b(decidido|decidimos|fechado|combinado|aprovado|definido|definimos|escolhemos|ficou acordado)\b",
    re.IGNORECASE)


def generate_heuristic_minutes(
    segments: Sequence[SpeakerSegment],
    title: str,
    duration_minutes: float,
    meeting_date: str,
    context: Optional[MeetingContext] = None,
    reason: str = "LLM indisponível",
) -> MeetingMinutes:
    context = context or MeetingContext()
    decisions: List[Decision] = []
    actions: List[ActionItem] = []
    for seg in segments:
        if DECISION_PATTERN.search(seg.text):
            decisions.append(Decision(description=f"{seg.speaker_id}: {seg.text}", evidence=[seg.id]))
        elif ACTION_PATTERN.search(seg.text):
            actions.append(ActionItem(task=seg.text, owner=seg.speaker_id, evidence=[seg.id]))

    speakers = list(dict.fromkeys(s.speaker_id for s in segments))
    objectives = []
    if context.objective:
        objectives.append(Objective(description=context.objective, origin="informado", status="indefinido",
                                    notes="Não avaliado no modo de contingência."))

    return MeetingMinutes(
        title=title,
        date=meeting_date,
        duration_minutes=duration_minutes,
        participants=speakers,
        meeting_type=context.meeting_type,
        objectives=objectives,
        executive_summary=(
            f"Reunião de {duration_minutes:.0f} minutos com {len(speakers)} participante(s). "
            "Ata gerada automaticamente por palavras-chave, sem análise semântica — revise os itens abaixo."
        ),
        main_topics=[TopicItem(
            title="Transcrição integral",
            discussion="Os temas tratados estão registrados na transcrição completa.",
            evidence=[],
        )],
        decisions=decisions[:15],
        action_items=actions[:20],
        source="heuristic",
        strategy="heuristic",
        warnings=[f"Ata gerada sem LLM ({reason}). Itens extraídos por palavras-chave podem conter falsos positivos."],
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
