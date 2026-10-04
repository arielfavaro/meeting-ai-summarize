"""
Correções manuais de locutores (caso de uso "editar locutores"):
- mesclar locutores que a diarização separou indevidamente;
- reatribuir trechos a outro locutor (ou a um locutor novo = "dividir");
- preservar nomes ao re-diarizar, casando locutores antigos e novos por sobreposição de tempo.

Tudo opera sobre IDs estáveis; a ata é atualizada trocando rótulos, sem chamar o LLM.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

from backend.models.schemas import MeetingDetail, SpeakerSegment
from backend.services.speakers import relabel_minutes, sync_display_names

LABEL_NUMBER = re.compile(r"^Locutor (\d+)$")


class SpeakerEditError(ValueError):
    pass


def next_speaker_label(meeting: MeetingDetail) -> str:
    used = [int(m.group(1)) for sid in {*meeting.speaker_ids(), *meeting.speaker_map} if (m := LABEL_NUMBER.match(sid))]
    return f"Locutor {max(used, default=0) + 1}"


def merge_speakers(meeting: MeetingDetail, source_ids: Sequence[str], target_id: str) -> MeetingDetail:
    ids = set(meeting.speaker_ids())
    sources = [s for s in dict.fromkeys(source_ids) if s != target_id]
    if target_id not in ids:
        raise SpeakerEditError(f"Locutor '{target_id}' não existe nesta reunião.")
    missing = [s for s in sources if s not in ids]
    if missing or not sources:
        raise SpeakerEditError(f"Locutores inválidos para mesclar: {', '.join(missing) or 'nenhum'}.")

    for seg in meeting.segments:
        if seg.speaker_id in sources:
            seg.speaker_id = target_id

    # O destino herda um nome real de uma das origens se ainda não tiver um.
    if meeting.speaker_name_sources.get(target_id, "default") == "default":
        for src in sources:
            if meeting.speaker_name_sources.get(src) in ("user", "llm"):
                meeting.speaker_map[target_id] = meeting.speaker_map[src]
                meeting.speaker_name_sources[target_id] = meeting.speaker_name_sources[src]
                break
    for src in sources:
        meeting.speaker_map.pop(src, None)
        meeting.speaker_name_sources.pop(src, None)

    if meeting.summary:
        meeting.summary = relabel_minutes(meeting.summary, {src: target_id for src in sources})
    return sync_display_names(meeting)


def reassign_segments(meeting: MeetingDetail, segment_ids: Sequence[int],
                      speaker_id: Optional[str] = None) -> Tuple[MeetingDetail, str]:
    by_id = {s.id: s for s in meeting.segments}
    unknown = [i for i in segment_ids if i not in by_id]
    if unknown:
        raise SpeakerEditError(f"Trechos inexistentes: {unknown}.")
    if speaker_id is None:
        speaker_id = next_speaker_label(meeting)
        meeting.speaker_map[speaker_id] = speaker_id
        meeting.speaker_name_sources[speaker_id] = "default"
    elif speaker_id not in meeting.speaker_map and speaker_id not in meeting.speaker_ids():
        raise SpeakerEditError(f"Locutor '{speaker_id}' não existe nesta reunião.")

    for i in segment_ids:
        by_id[i].speaker_id = speaker_id
    return sync_display_names(meeting), speaker_id


def _speaker_durations(segments: Sequence[SpeakerSegment]) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for s in segments:
        out.setdefault(s.speaker_id, []).append((s.start, s.end))
    return out


def _overlap(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    total, i, j = 0.0, 0, 0
    a, b = sorted(a), sorted(b)
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def carry_speaker_names(old: MeetingDetail, new_segments: Sequence[SpeakerSegment],
                        min_share: float = 0.5) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Após re-diarizar, os rótulos mudam. Casa cada locutor novo com o antigo de maior
    sobreposição de fala (pareamento guloso 1:1) e mantém os nomes já confirmados.
    """
    old_spans = _speaker_durations(old.segments)
    new_spans = _speaker_durations(new_segments)
    pairs = []
    for new_id, nspans in new_spans.items():
        new_total = sum(e - s for s, e in nspans) or 1e-6
        for old_id, ospans in old_spans.items():
            ov = _overlap(nspans, ospans)
            if ov / new_total >= min_share:
                pairs.append((ov, new_id, old_id))

    speaker_map = {sid: sid for sid in new_spans}
    sources = {sid: "default" for sid in new_spans}
    used_new, used_old = set(), set()
    for _, new_id, old_id in sorted(pairs, reverse=True):
        if new_id in used_new or old_id in used_old:
            continue
        used_new.add(new_id)
        used_old.add(old_id)
        if old.speaker_name_sources.get(old_id) in ("user", "llm"):
            speaker_map[new_id] = old.speaker_map[old_id]
            sources[new_id] = old.speaker_name_sources[old_id]
    return speaker_map, sources
