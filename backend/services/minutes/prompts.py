"""Carregamento dos prompts versionados (backend/prompts)."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.schemas import MeetingContext

PROMPT_VERSION = "minutes-v2"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _meeting_types() -> Dict[str, str]:
    return json.loads(_read("meeting_types.json"))


def system_prompt(kind: str, meeting_type: str = "geral") -> str:
    """kind: 'single' | 'extract' | 'reduce'."""
    template = _read(f"minutes_{kind}.md")
    guidance = _meeting_types().get(meeting_type, "")
    return (
        template.replace("{common_rules}", _read("common_rules.md").strip())
        .replace("{meeting_type_guidance}", guidance)
        .strip()
    )


def speakers_block(speaker_ids: List[str], speaker_map: Dict[str, str], sources: Dict[str, str]) -> str:
    lines = ["LOCUTORES (rótulos usados na transcrição):"]
    for sid in speaker_ids:
        name = speaker_map.get(sid)
        if name and name != sid and sources.get(sid) == "user":
            lines.append(f"- {sid} — nome confirmado: {name}")
        else:
            lines.append(f"- {sid} — nome desconhecido")
    return "\n".join(lines)


def context_block(
    context: MeetingContext,
    title: str,
    meeting_date: str,
    duration_minutes: float,
    custom_prompt: Optional[str] = None,
) -> str:
    lines = [
        "CONTEXTO DA REUNIÃO",
        f"- Título informado: {title}",
        f"- Data: {meeting_date}",
        f"- Duração: {duration_minutes:.0f} minutos",
    ]
    if context.objective:
        lines.append(f"- Objetivo/pauta informado pelo usuário: {context.objective}")
    if context.participants:
        lines.append(f"- Participantes esperados: {', '.join(context.participants)}")
    if context.glossary:
        lines.append(f"- Glossário (grafia correta de termos): {', '.join(context.glossary)}")
    extra = custom_prompt if custom_prompt is not None else context.custom_prompt
    if extra:
        lines.append(f"- Instruções adicionais do usuário: {extra}")
    return "\n".join(lines)
