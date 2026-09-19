from typing import List, Dict, Any
from backend.models.schemas import SpeakerSegment


def _find_best_speaker(start: float, end: float, diarization_segments: List[Dict[str, Any]]) -> str:
    """Encontra o locutor correspondente a um intervalo temporal com base na sobreposição máxima."""
    mid = (start + end) / 2.0
    best_speaker = "Locutor 1"
    max_overlap = 0.0

    for d_seg in diarization_segments:
        overlap_start = max(start, d_seg["start"])
        overlap_end = min(end, d_seg["end"])
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = d_seg["speaker"]

    if max_overlap <= 0.0:
        min_dist = float("inf")
        for d_seg in diarization_segments:
            d_mid = (d_seg["start"] + d_seg["end"]) / 2.0
            dist = abs(mid - d_mid)
            if dist < min_dist:
                min_dist = dist
                best_speaker = d_seg["speaker"]

    return best_speaker


def align_transcription_with_diarization(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[SpeakerSegment]:
    """
    Combina os segmentos de texto transcritos com os intervalos de tempo dos locutores.
    Se o segmento contiver timestamps por palavra ('words'), permite a separação cirúrgica
    quando houver alternância de locutores no meio de uma mesma frase.
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

    for t_seg in transcription_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        words = t_seg.get("words", [])

        # Se houver timestamps individuais de palavras, faz divisão fina
        if words and len(words) > 1:
            current_spk = None
            cur_words = []
            cur_start = None
            cur_end = None

            for w in words:
                w_text = w["word"]
                w_start = w["start"]
                w_end = w["end"]
                w_spk = _find_best_speaker(w_start, w_end, diarization_segments)

                if current_spk is None:
                    current_spk = w_spk
                    cur_start = w_start
                    cur_end = w_end
                    cur_words.append(w_text)
                elif w_spk == current_spk:
                    cur_end = w_end
                    cur_words.append(w_text)
                else:
                    # Troca de locutor na mesma frase
                    aligned.append(
                        SpeakerSegment(
                            id=len(aligned) + 1,
                            start=cur_start,
                            end=cur_end,
                            speaker=current_spk,
                            text=" ".join(cur_words).strip()
                        )
                    )
                    current_spk = w_spk
                    cur_start = w_start
                    cur_end = w_end
                    cur_words = [w_text]

            if cur_words:
                aligned.append(
                    SpeakerSegment(
                        id=len(aligned) + 1,
                        start=cur_start,
                        end=cur_end,
                        speaker=current_spk,
                        text=" ".join(cur_words).strip()
                    )
                )
        else:
            best_spk = _find_best_speaker(t_start, t_end, diarization_segments)
            aligned.append(
                SpeakerSegment(
                    id=len(aligned) + 1,
                    start=t_start,
                    end=t_end,
                    speaker=best_spk,
                    text=t_seg["text"].strip()
                )
            )

    # Mesclar segmentos consecutivos idênticos do mesmo orador se a pausa for < 0.8s
    merged: List[SpeakerSegment] = []
    for seg in aligned:
        if merged and merged[-1].speaker == seg.speaker and (seg.start - merged[-1].end) < 0.8:
            merged[-1].end = seg.end
            merged[-1].text = f"{merged[-1].text} {seg.text}".strip()
        else:
            seg.id = len(merged) + 1
            merged.append(seg)

    return merged


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
