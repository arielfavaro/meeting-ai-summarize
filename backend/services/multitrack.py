"""
Ajustes específicos de gravações multi-faixa (OBS/Zoom/Meet com faixa de microfone separada).

1. Canal de microfone individual: a faixa do microfone tem praticamente uma voz só, mas o
   diarizador costuma "achar" 2-3 locutores ali (respiração, ruído, vazamento do alto-falante).
   Se um locutor domina a faixa, ela é tratada como um locutor único.
2. Eco entre faixas: sem fone de ouvido, o microfone capta as vozes que saem do alto-falante,
   e a mesma fala aparece transcrita nas duas faixas. Falas simultâneas com texto quase
   idêntico em faixas diferentes são deduplicadas.
"""
import bisect
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Set, Tuple

from backend.models.schemas import SpeakerSegment
from backend.services.diarization.base import Turn


def speaker_shares(turns: Sequence[Turn]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for t in turns:
        totals[t["speaker"]] = totals.get(t["speaker"], 0.0) + max(0.0, t["end"] - t["start"])
    total = sum(totals.values()) or 1.0
    return {spk: dur / total for spk, dur in totals.items()}


def collapse_dominant_speaker(turns: List[Turn], min_share: float) -> Tuple[List[Turn], Optional[float]]:
    """
    Se um locutor responde por `min_share` (ex.: 80%) ou mais da fala da faixa, todos os turnos
    viram dele. Retorna (turnos, participação do dominante) — participação None = não colapsou.
    """
    if not turns or min_share <= 0:
        return turns, None
    shares = speaker_shares(turns)
    if len(shares) <= 1:
        return turns, None
    dominant, share = max(shares.items(), key=lambda kv: kv[1])
    if share < min_share:
        return turns, None
    return [{**t, "speaker": dominant} for t in turns], share


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text)).strip()


def _overlap_ratio(a: SpeakerSegment, b: SpeakerSegment) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    shorter = min(a.end - a.start, b.end - b.start)
    return inter / shorter if inter > 0 and shorter > 0 else 0.0


def remove_cross_track_echo(
    per_track: List[List[SpeakerSegment]],
    single_speaker_tracks: Set[int],
    min_overlap: float = 0.5,
    min_similarity: float = 0.75,
    min_chars: int = 12,
) -> Tuple[List[List[SpeakerSegment]], int]:
    """
    Remove a cópia "eco" de falas que aparecem em duas faixas ao mesmo tempo com o mesmo texto.
    Preferência: manter a cópia da faixa com vários locutores (a faixa da chamada) e descartar a
    do canal de microfone individual; sem essa pista, mantém a cópia com mais texto.
    """
    drop: List[Set[int]] = [set() for _ in per_track]
    starts = [[s.start for s in segs] for segs in per_track]

    for ti, segs in enumerate(per_track):
        for si, seg in enumerate(segs):
            if si in drop[ti] or len(_norm(seg.text)) < min_chars:
                continue
            for tj in range(ti + 1, len(per_track)):
                other = per_track[tj]
                hi = bisect.bisect_left(starts[tj], seg.end)
                for sj in range(max(0, hi - 50), hi):
                    cand = other[sj]
                    if sj in drop[tj] or cand.end <= seg.start or _overlap_ratio(seg, cand) < min_overlap:
                        continue
                    if SequenceMatcher(None, _norm(seg.text), _norm(cand.text)).ratio() < min_similarity:
                        continue
                    if ti in single_speaker_tracks and tj not in single_speaker_tracks:
                        loser = (ti, si)
                    elif tj in single_speaker_tracks and ti not in single_speaker_tracks:
                        loser = (tj, sj)
                    else:
                        loser = (ti, si) if len(seg.text) < len(cand.text) else (tj, sj)
                    drop[loser[0]].add(loser[1])
                    if loser == (ti, si):
                        break
                if si in drop[ti]:
                    break

    removed = sum(len(d) for d in drop)
    cleaned = [[s for i, s in enumerate(segs) if i not in drop[t]] for t, segs in enumerate(per_track)]
    return cleaned, removed
