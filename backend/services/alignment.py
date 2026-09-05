from typing import List, Dict, Any
from backend.models.schemas import SpeakerSegment


def align_transcription_with_diarization(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[SpeakerSegment]:
    """
    Combina os segmentos de texto transcritos com os intervalos de tempo dos locutores.
    Para cada trecho de fala, calcula a maior sobreposição temporal com os intervalos do diarizador.
    """
    if not diarization_segments:
        # Sem diarização: atribui tudo ao Locutor 1
        return [
            SpeakerSegment(
                id=i + 1,
                start=seg["start"],
                end=seg["end"],
                speaker="Locutor 1",
                text=seg["text"]
            )
            for i, seg in enumerate(transcription_segments)
        ]

    aligned: List[SpeakerSegment] = []

    for i, t_seg in enumerate(transcription_segments):
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        t_mid = (t_start + t_end) / 2.0

        best_speaker = "Locutor 1"
        max_overlap = 0.0

        # Procura intervalo de orador com maior interseção temporal
        for d_seg in diarization_segments:
            d_start = d_seg["start"]
            d_end = d_seg["end"]

            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_seg["speaker"]

        # Se não houve interseção direta, seleciona o orador mais próximo no tempo
        if max_overlap <= 0.0:
            min_dist = float("inf")
            for d_seg in diarization_segments:
                d_mid = (d_seg["start"] + d_seg["end"]) / 2.0
                dist = abs(t_mid - d_mid)
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = d_seg["speaker"]

        aligned.append(
            SpeakerSegment(
                id=i + 1,
                start=t_start,
                end=t_end,
                speaker=best_speaker,
                text=t_seg["text"]
            )
        )

    # Opcional: mesclar segmentos consecutivos idênticos do mesmo orador se a pausa for < 1s
    merged: List[SpeakerSegment] = []
    for seg in aligned:
        if merged and merged[-1].speaker == seg.speaker and (seg.start - merged[-1].end) < 1.0:
            merged[-1].end = seg.end
            merged[-1].text = f"{merged[-1].text} {seg.text}".strip()
        else:
            seg.id = len(merged) + 1
            merged.append(seg)

    return merged
