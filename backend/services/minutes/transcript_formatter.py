"""Formata a transcrição para o LLM com IDs referenciáveis e faz o fatiamento (chunking)."""
from typing import List, Sequence

from backend.models.schemas import SpeakerSegment
from backend.services.minutes.renderer import format_timestamp

# Português fica em ~3 caracteres/token nos tokenizadores de Gemma/Qwen/Llama.
CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def format_segment(seg: SpeakerSegment) -> str:
    # Usa SEMPRE o ID estável do locutor: a ata referencia rótulos e os nomes
    # são resolvidos depois, na apresentação.
    return f"[#{seg.id} {format_timestamp(seg.start)}] {seg.speaker_id}: {seg.text}"


def format_transcript(segments: Sequence[SpeakerSegment]) -> str:
    return "\n".join(format_segment(s) for s in segments)


def chunk_segments(
    segments: Sequence[SpeakerSegment],
    max_tokens: int,
    overlap_segments: int = 3,
) -> List[List[SpeakerSegment]]:
    """
    Divide a transcrição em blocos de até `max_tokens`, respeitando limites de fala
    e com pequena sobreposição para não perder contexto na fronteira.
    """
    if not segments:
        return []
    chunks: List[List[SpeakerSegment]] = []
    current: List[SpeakerSegment] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = estimate_tokens(format_segment(seg)) + 1
        if current and current_tokens + seg_tokens > max_tokens:
            chunks.append(current)
            tail = current[-overlap_segments:] if overlap_segments > 0 else []
            current = list(tail)
            current_tokens = sum(estimate_tokens(format_segment(s)) + 1 for s in current)
        current.append(seg)
        current_tokens += seg_tokens

    if current:
        chunks.append(current)
    return chunks
