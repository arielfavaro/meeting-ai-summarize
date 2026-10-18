"""
Biblioteca de vozes conhecidas (biometria de voz guardada SÓ no SQLite local).

- Cadastro: quando o usuário dá nome a um locutor, a voz daquele locutor (média dos trechos
  dele na reunião) é somada à assinatura da pessoa — várias reuniões refinam a assinatura.
- Identificação: em uma reunião nova, cada locutor é comparado com as assinaturas; o nome só
  é aplicado com similaridade acima do limiar, com folga sobre a 2ª melhor pessoa e no
  máximo uma pessoa por locutor (pareamento 1:1).
"""
import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from backend.services.voice.embedder import SpeakerVoice, cosine, unit

# Peso máximo de uma única reunião no cadastro: uma reunião longa não "apaga" as anteriores.
MAX_ENROLL_SECONDS = 300.0


def name_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


@dataclass
class Voiceprint:
    id: str
    name: str
    vector: np.ndarray
    seconds: float = 0.0       # fala acumulada (com teto por reunião)
    samples: int = 0           # quantas reuniões contribuíram
    created_at: str = ""
    updated_at: str = ""

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "seconds": round(self.seconds, 1), "samples": self.samples,
                "created_at": self.created_at, "updated_at": self.updated_at}


@dataclass
class VoiceMatch:
    speaker_id: str
    voiceprint: Voiceprint
    score: float
    runner_up: float


def enroll_vector(existing: Optional[Voiceprint], voice: SpeakerVoice, name: str, now: str) -> Voiceprint:
    weight = min(voice.seconds, MAX_ENROLL_SECONDS)
    if existing is None:
        return Voiceprint(id=str(uuid.uuid4()), name=name, vector=unit(voice.vector), seconds=weight, samples=1,
                          created_at=now, updated_at=now)
    old_weight = max(existing.seconds, 1e-3)
    vector = unit(existing.vector * old_weight + unit(voice.vector) * weight)
    return Voiceprint(id=existing.id, name=name or existing.name, vector=vector, seconds=old_weight + weight,
                      samples=existing.samples + 1, created_at=existing.created_at, updated_at=now)


def match_voices(voices: Dict[str, SpeakerVoice], prints: List[Voiceprint], threshold: float,
                 margin: float = 0.05, min_seconds: float = 5.0) -> List[VoiceMatch]:
    """Pareamento guloso 1:1 pelos maiores escores, com limiar e folga sobre o 2º colocado."""
    candidates = []
    for sid, voice in voices.items():
        if voice is None or voice.seconds < min_seconds or not prints:
            continue
        scores = sorted(((cosine(voice.vector, p.vector), p) for p in prints), key=lambda x: x[0], reverse=True)
        best, runner = scores[0], (scores[1][0] if len(scores) > 1 else -1.0)
        if best[0] >= threshold and best[0] - runner >= margin:
            candidates.append(VoiceMatch(sid, best[1], best[0], runner))

    matches: List[VoiceMatch] = []
    used_speakers, used_prints = set(), set()
    for m in sorted(candidates, key=lambda m: m.score, reverse=True):
        if m.speaker_id in used_speakers or m.voiceprint.id in used_prints:
            continue
        used_speakers.add(m.speaker_id)
        used_prints.add(m.voiceprint.id)
        matches.append(m)
    return matches
