"""
Extração de assinaturas de voz por locutor.

Para cada locutor, escolhe os trechos de fala mais longos (fatiados em janelas de até
`max_window` s), extrai um embedding por janela e faz a média ponderada pela duração. O
resultado é um vetor unitário: comparar dois locutores = similaridade de cosseno.

A parte de seleção de janelas e média é pura (numpy) e testável; o modelo neural entra
como uma função `encode(janelas) -> matriz` injetável.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
Span = Tuple[float, float]
EncodeFn = Callable[[List[np.ndarray]], np.ndarray]


@dataclass
class SpeakerVoice:
    vector: np.ndarray   # unitário (norma 1)
    seconds: float       # quantos segundos de fala sustentam o vetor

    def to_json(self) -> dict:
        return {"vector": [round(float(x), 6) for x in self.vector], "seconds": round(self.seconds, 2)}

    @classmethod
    def from_json(cls, data: dict) -> "SpeakerVoice":
        return cls(np.asarray(data["vector"], dtype=np.float32), float(data["seconds"]))


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def combine(voices: Iterable[SpeakerVoice]) -> Optional[SpeakerVoice]:
    """Média ponderada pela duração (ex.: dois rótulos fundidos viram uma voz só)."""
    voices = [v for v in voices if v is not None]
    if not voices:
        return None
    total = sum(v.seconds for v in voices) or 1.0
    vector = sum(v.vector * (v.seconds / total) for v in voices)
    return SpeakerVoice(unit(vector), sum(v.seconds for v in voices))


def select_windows(spans: Sequence[Span], min_window: float = 1.0, max_window: float = 8.0,
                   max_total: float = 180.0, max_windows: int = 40) -> List[Span]:
    """
    Janelas de fala para o embedding: os trechos mais longos primeiro (mais limpos), fatiados
    em pedaços de até `max_window` s, respeitando um orçamento total (custo previsível).
    """
    windows: List[Span] = []
    total = 0.0
    for start, end in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        duration = end - start
        if duration < min_window:
            break  # ordenados do maior para o menor: os próximos também são curtos
        pieces = int(np.ceil(duration / max_window))  # partes iguais, cada uma entre min e max
        step = duration / pieces
        for k in range(pieces):
            windows.append((start + k * step, start + (k + 1) * step))
            total += step
            if total >= max_total or len(windows) >= max_windows:
                return windows
    return windows


def spans_by_speaker(turns: Iterable[dict], key: str = "speaker") -> Dict[str, List[Span]]:
    out: Dict[str, List[Span]] = {}
    for t in turns:
        out.setdefault(t[key], []).append((float(t["start"]), float(t["end"])))
    return out


def embed_speakers(audio: np.ndarray, spans: Dict[str, List[Span]], encode: EncodeFn,
                   sr: int = SAMPLE_RATE, min_seconds: float = 3.0, **window_opts) -> Dict[str, SpeakerVoice]:
    """Um `SpeakerVoice` por locutor com fala suficiente (`min_seconds`); os demais ficam de fora."""
    result: Dict[str, SpeakerVoice] = {}
    for speaker, speaker_spans in spans.items():
        windows = select_windows(speaker_spans, **window_opts)
        seconds = sum(e - s for s, e in windows)
        if seconds < min_seconds:
            continue
        chunks = [audio[int(s * sr):int(e * sr)] for s, e in windows]
        chunks = [c for c in chunks if len(c) >= int(0.5 * sr)]
        if not chunks:
            continue
        vectors = np.asarray(encode(chunks), dtype=np.float32)
        weights = np.asarray([len(c) for c in chunks], dtype=np.float32)
        vectors = np.stack([unit(v) for v in vectors])
        result[speaker] = SpeakerVoice(unit((vectors * weights[:, None]).sum(axis=0)), seconds)
    return result


def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


class EcapaEncoder:
    """SpeechBrain ECAPA-TDNN (VoxCeleb), o mesmo modelo local já usado pelo motor SpeechBrain."""

    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size

    @staticmethod
    def unavailable_reason() -> Optional[str]:
        from backend.config import settings
        from backend.services.diarization.base import has_module
        for module in ("torch", "speechbrain"):
            if not has_module(module):
                return f"biblioteca '{module}' não instalada"
        model_dir = settings.speechbrain_model_dir
        if not (model_dir / "hyperparams.yaml").exists():
            return f"modelo SpeechBrain não baixado em {model_dir}"
        return None

    def __call__(self, chunks: List[np.ndarray]) -> np.ndarray:
        import torch
        from backend.services.diarization.local_engines import get_speechbrain_classifier

        classifier = get_speechbrain_classifier()
        out = []
        with torch.no_grad():
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                longest = max(len(c) for c in batch)
                padded = np.zeros((len(batch), longest), dtype=np.float32)
                for j, c in enumerate(batch):
                    padded[j, :len(c)] = c
                rel_lens = torch.tensor([len(c) / longest for c in batch], dtype=torch.float32)
                emb = classifier.encode_batch(torch.from_numpy(padded).to(classifier.device),
                                              wav_lens=rel_lens.to(classifier.device))
                out.append(emb.squeeze(1).cpu().numpy())
        return np.vstack(out)

    @staticmethod
    def release() -> None:
        from backend.services.diarization.local_engines import release_speechbrain_classifier
        release_speechbrain_classifier()
