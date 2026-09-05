import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import settings

logger = logging.getLogger(__name__)


class TranscriptionService:
    _models: Dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_size: str = "small", device: Optional[str] = None, compute_type: Optional[str] = None):
        """Carrega ou reutiliza instância do modelo Faster-Whisper."""
        device = device or settings.WHISPER_DEVICE
        compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE

        # Se for CPU e compute_type for float16, mudar para int8
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        cache_key = f"{model_size}_{device}_{compute_type}"
        if cache_key not in cls._models:
            logger.info(f"Carregando modelo Faster-Whisper: {model_size} (device={device}, compute_type={compute_type})...")
            from faster_whisper import WhisperModel
            cls._models[cache_key] = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(settings.MODELS_CACHE_DIR)
            )
            logger.info(f"Modelo Faster-Whisper {model_size} carregado com sucesso!")
        return cls._models[cache_key]

    @classmethod
    def transcribe(
        cls,
        audio_path: Path,
        model_size: str = "small",
        language: str = "pt",
        beam_size: int = 5,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Executa transcrição em áudio com detecção de idioma e timestamps.
        Retorna lista de segmentos com start, end e text.
        """
        logger.info(f"Iniciando transcrição de {audio_path.name} com modelo {model_size} (idioma={language})...")
        model = cls.get_model(model_size=model_size)

        segments, info = model.transcribe(
            str(audio_path),
            language=language if language != "auto" else None,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False
        )

        logger.info(f"Transcrição detectou idioma: {info.language} (probabilidade={info.language_probability:.2f})")

        total_duration = getattr(info, "duration", 0.0) or 0.0
        transcribed_segments = []
        for s in segments:
            text = s.text.strip()
            if text:
                transcribed_segments.append({
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": text
                })
            if progress_callback and total_duration > 0:
                progress_callback(s.end, total_duration)

        return transcribed_segments
