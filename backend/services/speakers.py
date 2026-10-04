"""
Identidade de locutores: IDs estáveis x nomes de exibição.

- `speaker_id` ("Locutor 1") nunca muda depois da diarização.
- `speaker_map` traduz speaker_id -> nome de exibição ("Ariel").
- Textos da ata usam os rótulos estáveis e são resolvidos na apresentação,
  de modo que renomear um locutor atualiza ata, transcrição e exportações
  sem precisar chamar o LLM novamente.
"""
import re
from typing import Dict, Iterable, List

from backend.models.schemas import MeetingDetail, MeetingMinutes

LABEL_PATTERN = re.compile(r"\bLocutor \d+\b")


def resolve_labels(text: str, speaker_map: Dict[str, str]) -> str:
    """Substitui rótulos estáveis ("Locutor 2") pelo nome de exibição."""
    if not text or not speaker_map:
        return text
    return LABEL_PATTERN.sub(lambda m: speaker_map.get(m.group(0)) or m.group(0), text)


def is_label(value: str) -> bool:
    return bool(LABEL_PATTERN.fullmatch((value or "").strip()))


def normalize_legacy_meeting(meeting: MeetingDetail) -> MeetingDetail:
    """
    Reuniões antigas gravavam o nome de exibição no lugar do ID do locutor
    (ex.: segment.speaker == "Ariel" e speaker_map == {"Locutor 1": "Ariel"}).
    Recupera o ID estável a partir do mapa reverso.
    """
    reverse = {name: sid for sid, name in meeting.speaker_map.items() if name and name != sid}
    for seg in meeting.segments:
        if seg.speaker_id == seg.speaker and seg.speaker in reverse:
            seg.speaker_id = reverse[seg.speaker]
    for sid in meeting.speaker_ids():
        meeting.speaker_map.setdefault(sid, sid)
        meeting.speaker_name_sources.setdefault(
            sid, "user" if meeting.speaker_map[sid] != sid else "default"
        )
    return meeting


def sync_display_names(meeting: MeetingDetail) -> MeetingDetail:
    """Recalcula o campo desnormalizado `speaker` de cada segmento (in-place)."""
    for seg in meeting.segments:
        seg.speaker = meeting.display_name(seg.speaker_id)
    return meeting


def _resolve_minutes(summary: MeetingMinutes, speaker_map: Dict[str, str], participants: List[str]) -> MeetingMinutes:
    s = _map_minutes_text(summary, speaker_map, map_suggestions=False)
    s.participants = participants
    return s


def present_meeting(meeting: MeetingDetail) -> MeetingDetail:
    """
    Retorna uma cópia pronta para exibição/exportação: nomes reais resolvidos
    na transcrição e na ata, e o Markdown da ata renderizado a partir dos dados
    estruturados.
    """
    from backend.services.minutes.renderer import render_minutes_markdown

    view = meeting.model_copy(deep=True)
    sync_display_names(view)
    if view.summary:
        participants = [view.display_name(sid) for sid in view.speaker_ids()]
        view.summary = _resolve_minutes(view.summary, view.speaker_map, participants)
        view.summary.raw_markdown = render_minutes_markdown(view)
    return view


def unique_display_names(meeting: MeetingDetail) -> Iterable[str]:
    return [meeting.display_name(sid) for sid in meeting.speaker_ids()]


def relabel_minutes(summary: MeetingMinutes, mapping: Dict[str, str]) -> MeetingMinutes:
    """Troca rótulos estáveis na ata (ex.: ao mesclar "Locutor 3" em "Locutor 1")."""
    if not summary or not mapping:
        return summary
    s = _map_minutes_text(summary, mapping, map_suggestions=True)
    s.participants = list(dict.fromkeys(mapping.get(p, p) for p in s.participants))
    return s


def _map_minutes_text(summary: MeetingMinutes, mapping: Dict[str, str], *, map_suggestions: bool) -> MeetingMinutes:
    """Aplica o mapeamento de rótulos em todos os textos da ata (cópia)."""
    r = lambda t: resolve_labels(t, mapping) if isinstance(t, str) else t  # noqa: E731
    s = summary.model_copy(deep=True)
    s.title, s.executive_summary = r(s.title), r(s.executive_summary)
    for o in s.objectives:
        o.description, o.notes = r(o.description), r(o.notes)
    for t in s.main_topics:
        t.title, t.discussion, t.conclusions = r(t.title), r(t.discussion), r(t.conclusions)
    for d in s.decisions:
        d.description = r(d.description)
    for p in [*s.open_points, *s.risks]:
        p.description = r(p.description)
    for a in s.action_items:
        a.task, a.owner = r(a.task), r(a.owner)
    if map_suggestions:
        for sug in s.speaker_suggestions:
            sug.speaker_id = mapping.get(sug.speaker_id, sug.speaker_id)
    return s
