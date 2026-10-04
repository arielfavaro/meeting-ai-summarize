"""Motores locais legados (SpeechBrain ECAPA e acústico) adaptados ao contrato DiarizationEngine."""
from pathlib import Path
from typing import List, Optional

from backend.services.diarization.base import LogCallback, Turn, has_module
from backend.services.diarization.local_engines import LocalDiarizationEngines, release_speechbrain_classifier


class SpeechBrainEngine:
    name = "speechbrain"
    label = "SpeechBrain ECAPA-TDNN"

    def __init__(self) -> None:
        self._impl = LocalDiarizationEngines()

    def unavailable_reason(self) -> Optional[str]:
        for module in ("torch", "speechbrain", "librosa", "sklearn"):
            if not has_module(module):
                return f"biblioteca '{module}' não instalada"
        return None

    def diarize(self, audio_path: Path, min_speakers: Optional[int], max_speakers: Optional[int],
                log: LogCallback) -> List[Turn]:
        return self._impl._diarize_speechbrain(audio_path, min_speakers, max_speakers, log_callback=log)

    def release(self) -> None:
        release_speechbrain_classifier()


class AcousticEngine:
    """Último recurso: MFCC + pitch + clustering. Não depende de modelos baixados."""
    name = "acoustic"
    label = "motor acústico (Librosa)"

    def __init__(self) -> None:
        self._impl = LocalDiarizationEngines()

    def unavailable_reason(self) -> Optional[str]:
        for module in ("librosa", "sklearn"):
            if not has_module(module):
                return f"biblioteca '{module}' não instalada"
        return None

    def diarize(self, audio_path: Path, min_speakers: Optional[int], max_speakers: Optional[int],
                log: LogCallback) -> List[Turn]:
        return self._impl._diarize_local(audio_path, min_speakers, max_speakers)

    def release(self) -> None:
        pass
