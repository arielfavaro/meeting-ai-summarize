"""
Alinhamento transcrição × diarização com atribuição de locutor POR FRASE (estilo WhisperX).

Problema da atribuição palavra a palavra: uma única palavra que cai na borda de um turno
da diarização parte a frase em dois locutores. Aqui:
1. cada palavra recebe o locutor de maior sobreposição (linha do tempo com busca binária);
2. as palavras são agrupadas em frases (pontuação, pausas longas, ou pausa + troca de locutor);
3. cada frase vai para o locutor dominante (voto ponderado pela duração), exceto quando
   há uma interrupção real (trecho contínuo de outro locutor com várias palavras e duração);
4. "piscadas" curtas e de baixa confiança entre falas do mesmo locutor são absorvidas;
5. falas consecutivas do mesmo locutor são unidas (até um tamanho máximo, para que as
   evidências da ata continuem apontando trechos específicos).
"""
import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.models.schemas import SpeakerSegment

SENTENCE_END = re.compile(r"[.?!…]+[\"')\]]*$")


class SpeakerTimeline:
    """Consulta 'quem fala neste intervalo' em O(log n) sobre os turnos da diarização."""

    def __init__(self, turns: Sequence[Dict[str, Any]]):
        self.turns = sorted(turns, key=lambda t: (t["start"], t["end"]))
        self.starts = [t["start"] for t in self.turns]
        self.max_end: List[float] = []
        running = float("-inf")
        for t in self.turns:
            running = max(running, t["end"])
            self.max_end.append(running)

    def votes(self, start: float, end: float) -> Dict[str, float]:
        """Sobreposição (s) de cada locutor com [start, end]."""
        out: Dict[str, float] = {}
        i = bisect.bisect_left(self.starts, end) - 1
        while i >= 0 and self.max_end[i] > start:
            t = self.turns[i]
            overlap = min(end, t["end"]) - max(start, t["start"])
            if overlap > 0:
                out[t["speaker"]] = out.get(t["speaker"], 0.0) + overlap
            i -= 1
        return out

    def speaker_at(self, start: float, end: float) -> str:
        votes = self.votes(start, end)
        if votes:
            return max(votes.items(), key=lambda kv: kv[1])[0]
        return self.nearest(start, end)

    def nearest(self, start: float, end: float) -> str:
        if not self.turns:
            return "Locutor 1"
        mid = (start + end) / 2
        i = bisect.bisect_left(self.starts, mid)
        candidates = self.turns[max(0, i - 3): i + 2]

        def distance(t):
            if t["start"] <= mid <= t["end"]:
                return 0.0
            return min(abs(mid - t["start"]), abs(mid - t["end"]))
        return min(candidates, key=distance)["speaker"]


def _find_best_speaker(start: float, end: float, diarization_segments: List[Dict[str, Any]]) -> str:
    """Compatibilidade: locutor de maior sobreposição com o intervalo."""
    return SpeakerTimeline(diarization_segments).speaker_at(start, end)


@dataclass
class _Word:
    text: str
    start: float
    end: float
    speaker: str
    seg_index: int

    @property
    def duration(self) -> float:
        return max(0.05, self.end - self.start)


@dataclass
class _Turn:
    speaker: str
    words: List[_Word] = field(default_factory=list)
    share: float = 1.0  # fração (por duração) das palavras cuja voz bate com o locutor atribuído

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


@dataclass
class AlignmentConfig:
    sentence_gap: float = 1.0        # pausa que sempre encerra a frase
    change_gap: float = 0.4          # pausa que encerra a frase quando o locutor muda
    min_interrupt_words: int = 3     # interrupção real dentro de uma frase...
    min_interrupt_sec: float = 1.0   # ...precisa de várias palavras e duração mínima
    blip_max_sec: float = 0.6        # "piscada" que pode ser absorvida
    blip_max_words: int = 2
    blip_min_share: float = 0.7      # só absorve se a própria votação foi fraca
    merge_gap: float = 0.8           # une falas consecutivas do mesmo locutor
    max_segment_sec: float = 30.0    # tamanho máximo de uma fala unida


def _collect_words(transcription_segments, timeline: SpeakerTimeline) -> List[_Word]:
    words: List[_Word] = []
    for si, seg in enumerate(transcription_segments):
        seg_words = [w for w in (seg.get("words") or []) if str(w.get("word", "")).strip()]
        if seg_words:
            for w in seg_words:
                words.append(_Word(str(w["word"]).strip(), float(w["start"]), float(w["end"]),
                                   timeline.speaker_at(float(w["start"]), float(w["end"])), si))
        elif str(seg.get("text", "")).strip():
            words.append(_Word(seg["text"].strip(), float(seg["start"]), float(seg["end"]),
                               timeline.speaker_at(float(seg["start"]), float(seg["end"])), si))
    return words


def _split_sentences(words: List[_Word], cfg: AlignmentConfig) -> List[List[_Word]]:
    sentences: List[List[_Word]] = []
    current: List[_Word] = []
    for i, w in enumerate(words):
        current.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None:
            break
        gap = nxt.start - w.end
        speaker_changes = nxt.speaker != w.speaker
        boundary = (
            SENTENCE_END.search(w.text) is not None
            or gap >= cfg.sentence_gap
            or (gap >= cfg.change_gap and speaker_changes)
            or (nxt.seg_index != w.seg_index and speaker_changes)
        )
        if boundary:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def _runs(words: List[_Word]) -> List[List[_Word]]:
    runs: List[List[_Word]] = []
    for w in words:
        if runs and runs[-1][-1].speaker == w.speaker:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def _assign_sentence(sentence: List[_Word], cfg: AlignmentConfig) -> List[_Turn]:
    votes: Dict[str, float] = {}
    for w in sentence:
        votes[w.speaker] = votes.get(w.speaker, 0.0) + w.duration
    dominant = max(votes.items(), key=lambda kv: kv[1])[0]

    turns: List[_Turn] = []
    for run in _runs(sentence):
        speaker = run[0].speaker
        duration = run[-1].end - run[0].start
        is_interruption = (speaker != dominant and len(run) >= cfg.min_interrupt_words
                           and duration >= cfg.min_interrupt_sec)
        target = speaker if is_interruption else dominant
        if turns and turns[-1].speaker == target:
            turns[-1].words.extend(run)
        else:
            turns.append(_Turn(target, list(run)))

    for t in turns:
        total = sum(w.duration for w in t.words)
        t.share = sum(w.duration for w in t.words if w.speaker == t.speaker) / total if total else 1.0
    return turns


def _absorb_blips(turns: List[_Turn], cfg: AlignmentConfig) -> List[_Turn]:
    changed = True
    while changed:
        changed = False
        for i in range(1, len(turns) - 1):
            prev, cur, nxt = turns[i - 1], turns[i], turns[i + 1]
            if (prev.speaker == nxt.speaker != cur.speaker
                    and cur.end - cur.start <= cfg.blip_max_sec
                    and len(cur.words) <= cfg.blip_max_words
                    and cur.share < cfg.blip_min_share):
                prev.words.extend(cur.words + nxt.words)
                del turns[i:i + 2]
                changed = True
                break
    # A união de turnos do mesmo locutor fica para _to_segments (respeita o tamanho máximo).
    return turns


def _to_segments(turns: List[_Turn], cfg: AlignmentConfig) -> List[SpeakerSegment]:
    segments: List[SpeakerSegment] = []
    for t in turns:
        text = " ".join(w.text for w in t.words).strip()
        if not text:
            continue
        last = segments[-1] if segments else None
        if (last and last.speaker == t.speaker and t.start - last.end < cfg.merge_gap
                and t.end - last.start <= cfg.max_segment_sec):
            last.end = round(t.end, 2)
            last.text = f"{last.text} {text}".strip()
        else:
            segments.append(SpeakerSegment(id=len(segments) + 1, start=round(t.start, 2), end=round(t.end, 2),
                                           speaker=t.speaker, text=text))
    return segments


def align_transcription_with_diarization(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]],
    config: Optional[AlignmentConfig] = None,
) -> List[SpeakerSegment]:
    """Combina a transcrição (com timestamps por palavra) com os turnos da diarização."""
    cfg = config or AlignmentConfig()
    if not diarization_segments:
        return [
            SpeakerSegment(id=i + 1, start=seg["start"], end=seg["end"], speaker="Locutor 1", text=seg["text"].strip())
            for i, seg in enumerate(s for s in transcription_segments if str(s.get("text", "")).strip())
        ]

    timeline = SpeakerTimeline(diarization_segments)
    words = _collect_words(transcription_segments, timeline)
    if not words:
        return []

    turns: List[_Turn] = []
    for sentence in _split_sentences(words, cfg):
        turns.extend(_assign_sentence(sentence, cfg))
    turns = _absorb_blips(turns, cfg)
    return _to_segments(turns, cfg)


def merge_multitrack_segments(
    track_results: List[List[SpeakerSegment]]
) -> List[SpeakerSegment]:
    """
    Combina segmentos de múltiplas faixas de áudio ativas.
    1. Reatribui locutores de forma sequencial contínua (ex: Locutor 1 na faixa A, Locutores 2, 3, 4 na faixa B).
    2. Intercala todos os trechos cronologicamente por tempo de início (start).
    3. Renumera os IDs de 1 a N.
    """
    if not track_results:
        return []
    if len(track_results) == 1:
        return track_results[0]

    all_segments: List[SpeakerSegment] = []
    global_speaker_counter = 1

    for track_idx, segments in enumerate(track_results):
        if not segments:
            continue
        track_speaker_map = {}
        for seg in segments:
            local_spk = seg.speaker
            if local_spk not in track_speaker_map:
                track_speaker_map[local_spk] = f"Locutor {global_speaker_counter}"
                global_speaker_counter += 1

            new_seg = SpeakerSegment(
                id=0,
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                speaker=track_speaker_map[local_spk],
                text=seg.text.strip()
            )
            all_segments.append(new_seg)

    # Ordenar cronologicamente pelo timestamp de início (start)
    all_segments.sort(key=lambda s: (s.start, s.end))

    # Renumerar IDs sequencialmente de 1 a N
    for i, seg in enumerate(all_segments):
        seg.id = i + 1

    return all_segments
