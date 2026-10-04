"""
pyannote `speaker-diarization-community-1` 100% offline.

- O pipeline é carregado de um DIRETÓRIO LOCAL (baixado uma única vez pelo script
  `backend/scripts/download_models.py`). Em processamento não há token nem acesso à rede.
- Telemetria do pyannote desligada (PYANNOTE_METRICS_ENABLED=0 + set_telemetry_metrics(False)).
- O áudio é entregue em memória (waveform), sem decodificação externa.
- Usa a saída "exclusive" (um locutor por instante), feita para casar com timestamps de ASR.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.services.diarization.base import LogCallback, Turn, has_module, relabel_by_first_appearance

logger = logging.getLogger(__name__)

_PIPELINE = None
_PIPELINE_DIR: Optional[Path] = None

STEP_LABELS = {
    "segmentation": "segmentação de fala e sobreposição",
    "speaker_counting": "contagem de locutores",
    "embeddings": "assinaturas de voz (embeddings)",
    "discrete_diarization": "montagem da linha do tempo",
}


def _disable_telemetry() -> None:
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    try:
        from pyannote.audio.telemetry import set_telemetry_metrics
        set_telemetry_metrics(False)
    except Exception:  # versões sem o módulo de telemetria
        pass


class PyannoteEngine:
    name = "pyannote"
    label = "pyannote community-1"

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir or settings.pyannote_model_dir)

    def unavailable_reason(self) -> Optional[str]:
        if not has_module("pyannote.audio"):
            return "biblioteca 'pyannote.audio' não instalada"
        if not (self.model_dir / "config.yaml").exists():
            return f"modelo não baixado em {self.model_dir} (rode o script de download uma vez)"
        return None

    def _load(self):
        global _PIPELINE, _PIPELINE_DIR
        if _PIPELINE is not None and _PIPELINE_DIR == self.model_dir:
            return _PIPELINE
        _disable_telemetry()
        import torch
        from pyannote.audio import Pipeline

        logger.info("Carregando pyannote de %s (offline)...", self.model_dir)
        pipeline = Pipeline.from_pretrained(str(self.model_dir))
        if pipeline is None:
            raise RuntimeError(f"Não foi possível carregar o pipeline pyannote de {self.model_dir}")
        if torch.cuda.is_available() and settings.WHISPER_DEVICE == "cuda":
            pipeline.to(torch.device("cuda"))
        _PIPELINE, _PIPELINE_DIR = pipeline, self.model_dir
        return pipeline

    def diarize(self, audio_path: Path, min_speakers: Optional[int], max_speakers: Optional[int],
                log: LogCallback) -> List[Turn]:
        import numpy as np
        import soundfile as sf
        import torch

        pipeline = self._load()
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1)))[None, :]

        kwargs: Dict[str, Any] = {}
        if min_speakers and max_speakers and min_speakers == max_speakers:
            kwargs["num_speakers"] = min_speakers
        else:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

        seen_steps = set()

        def hook(step_name, step_artifact=None, file=None, total=None, completed=None, **_):
            if step_name not in seen_steps:
                seen_steps.add(step_name)
                log(f"pyannote: {STEP_LABELS.get(step_name, step_name)}...")

        output = pipeline({"waveform": waveform, "sample_rate": sr}, hook=hook, **kwargs)
        annotation = (getattr(output, "exclusive_speaker_diarization", None)
                      or getattr(output, "speaker_diarization", None)
                      or output)
        turns = [{"start": seg.start, "end": seg.end, "speaker": label}
                 for seg, _, label in annotation.itertracks(yield_label=True)]
        turns = relabel_by_first_appearance(turns)
        speakers = {t["speaker"] for t in turns}
        log(f"pyannote: {len(speakers)} locutor(es) em {len(turns)} turnos de fala.", "success")
        return turns

    def release(self) -> None:
        global _PIPELINE, _PIPELINE_DIR
        if _PIPELINE is not None:
            _PIPELINE, _PIPELINE_DIR = None, None
            from backend.services.diarization.local_engines import _empty_cuda_cache
            _empty_cuda_cache()
