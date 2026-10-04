"""
Serviço de diarização: escolhe o motor (Strategy) e aplica a cadeia de alternativas.

Ordem no modo "auto": pyannote community-1 (se o modelo estiver no disco) → SpeechBrain
→ acústico. Tudo roda localmente; nenhum motor faz chamada de rede em processamento.
"""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from backend.config import settings
from backend.services.diarization.base import DiarizationEngine, LogCallback, Turn
from backend.services.diarization.local import AcousticEngine, SpeechBrainEngine
from backend.services.diarization.pyannote_engine import PyannoteEngine

logger = logging.getLogger(__name__)

ENGINE_FACTORIES: Dict[str, Callable[[], DiarizationEngine]] = {
    "pyannote": PyannoteEngine,
    "speechbrain": SpeechBrainEngine,
    "acoustic": AcousticEngine,
}
AUTO_ORDER = ["pyannote", "speechbrain", "acoustic"]


def engine_chain(preferred: str) -> List[str]:
    """O motor pedido primeiro, seguido dos de menor qualidade como reserva."""
    preferred = (preferred or "auto").lower()
    if preferred not in ENGINE_FACTORIES:
        return list(AUTO_ORDER)
    return AUTO_ORDER[AUTO_ORDER.index(preferred):]


def configured_engine() -> str:
    if settings.ENABLE_PYANNOTE and settings.DIARIZATION_ENGINE == "auto":
        return "pyannote"
    return settings.DIARIZATION_ENGINE


class DiarizationService:
    def __init__(self, hf_token: Optional[str] = None, engine: Optional[str] = None,
                 engines: Optional[Dict[str, DiarizationEngine]] = None):
        # hf_token é aceito por compatibilidade e ignorado: o processamento é offline.
        self.preferred = engine or configured_engine()
        self._engines: Dict[str, DiarizationEngine] = engines or {}
        self.last_engine: Optional[str] = None

    def _engine(self, name: str) -> DiarizationEngine:
        if name not in self._engines:
            self._engines[name] = ENGINE_FACTORIES[name]()
        return self._engines[name]

    def status(self) -> Dict[str, Optional[str]]:
        """Disponibilidade de cada motor (para diagnóstico/health)."""
        return {name: self._engine(name).unavailable_reason() for name in AUTO_ORDER}

    def diarize(self, audio_path: Path, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None,
                log_callback: Optional[LogCallback] = None) -> List[Turn]:
        def log(message: str, level: str = "info") -> None:
            if log_callback:
                log_callback(message, level)

        chain = engine_chain(self.preferred)
        last_error: Optional[Exception] = None
        for name in chain:
            engine = self._engine(name)
            reason = engine.unavailable_reason()
            if reason:
                if name != chain[-1]:
                    level = "warning" if name == chain[0] and self.preferred != "auto" else "info"
                    log(f"{engine.label} indisponível ({reason}).", level)
                continue
            try:
                log(f"Separando locutores com {engine.label}...")
                turns = engine.diarize(audio_path, min_speakers, max_speakers, log)
                if not turns:
                    raise RuntimeError("nenhum turno de fala retornado")
                self.last_engine = name
                return turns
            except Exception as e:  # um motor quebrado não pode derrubar o processamento
                last_error = e
                logger.warning("Diarização com %s falhou: %s", name, e, exc_info=True)
                log(f"{engine.label} falhou ({e}). Tentando o próximo motor...", "warning")
        raise RuntimeError(f"Nenhum motor de diarização disponível: {last_error}")

    def release(self) -> None:
        for engine in self._engines.values():
            try:
                engine.release()
            except Exception:
                logger.debug("Falha ao liberar motor", exc_info=True)
