import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
from backend.config import settings

logger = logging.getLogger(__name__)

BASE_PROMPT_PT = (
    "Transcrição de reunião corporativa e técnica em português brasileiro, com pontuação correta, "
    "termos de negócios e acentuação adequada."
)
# O Whisper usa no máximo ~224 tokens de prompt; mantemos folga.
MAX_PROMPT_CHARS = 600


def build_initial_prompt(language: str, terms: Optional[Sequence[str]] = None) -> Optional[str]:
    """
    Monta o `initial_prompt` do Whisper com nomes de participantes e termos do glossário,
    o que melhora a grafia de nomes próprios, siglas e produtos.
    """
    base = BASE_PROMPT_PT if language == "pt" else ""
    terms = [t.strip() for t in (terms or []) if t and t.strip()]
    if terms:
        vocab = "Vocabulário: " + ", ".join(dict.fromkeys(terms)) + "."
        base = f"{base} {vocab}".strip()
    return base[:MAX_PROMPT_CHARS] or None


class TranscriptionService:
    _models: Dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_size: Optional[str] = None, device: Optional[str] = None, compute_type: Optional[str] = None):
        """
        Carrega ou reutiliza instância do modelo Faster-Whisper.
        Prioriza SEMPRE o carregamento 100% OFFLINE do disco local (sem checagem de rede).
        """
        model_size = model_size or settings.WHISPER_MODEL_SIZE
        device = device or settings.WHISPER_DEVICE
        compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE

        # Se for CPU e compute_type for float16, mudar para int8
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        cache_key = f"{model_size}_{device}_{compute_type}"
        if cache_key in cls._models:
            return cls._models[cache_key]

        logger.info(f"Inicializando Faster-Whisper: {model_size} (device={device}, compute={compute_type})...")
        from faster_whisper import WhisperModel

        def _try_load(dev: str, comp: str, local_only: bool):
            return WhisperModel(
                model_size,
                device=dev,
                compute_type=comp,
                download_root=str(settings.MODELS_CACHE_DIR),
                local_files_only=local_only
            )

        # 1. Tentativa 100% OFFLINE: lê direto do disco local sem fazer NENHUMA requisição de rede
        try:
            model = _try_load(device, compute_type, local_only=True)
            logger.info(f"✅ Modelo Faster-Whisper '{model_size}' carregado 100% LOCAL do disco ({device})!")
            cls._models[cache_key] = model
            return model
        except Exception as e_offline:
            logger.debug(f"Tentativa offline local de {model_size} em {device}: {e_offline}")

        # Se falhou offline no CUDA, tenta offline na CPU antes de qualquer conexão
        if device == "cuda":
            try:
                model = _try_load("cpu", "int8", local_only=True)
                logger.warning(f"⚠️ CUDA indisponível para {model_size}. Carregado 100% LOCAL na CPU (int8)!")
                cls._models[cache_key] = model
                return model
            except Exception:
                pass

        # 2. Modo offline estrito: nada de rede em processamento
        import os
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError(
                f"Modelo Whisper '{model_size}' não encontrado em {settings.MODELS_CACHE_DIR} (modo offline). "
                f"Baixe-o uma vez com: python -m backend.scripts.download_models --whisper {model_size}")

        # 3. Se o modelo nunca foi baixado para o disco, realiza o download inicial único
        logger.info(f"📥 Modelo '{model_size}' não encontrado no cache local. Baixando uma única vez para {settings.MODELS_CACHE_DIR}...")
        try:
            model = _try_load(device, compute_type, local_only=False)
            logger.info(f"✅ Download de '{model_size}' concluído e armazenado no disco local!")
            cls._models[cache_key] = model
            return model
        except Exception as e_down:
            if device == "cuda":
                logger.warning(f"Falha na GPU ({e_down}). Tentando download para execução em CPU...")
                return cls.get_model(model_size=model_size, device="cpu", compute_type="int8")
            raise e_down

    @classmethod
    def release(cls) -> None:
        """Descarrega os modelos Whisper da memória (libera VRAM antes do LLM)."""
        if not cls._models:
            return
        cls._models.clear()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Modelos Whisper liberados da memória.")

    @classmethod
    def transcribe(
        cls,
        audio_path: Path,
        model_size: Optional[str] = None,
        language: str = "pt",
        beam_size: int = 5,
        progress_callback=None,
        prompt_terms: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executa transcrição em áudio com detecção de idioma e timestamps.
        Retorna lista de segmentos com start, end e text.
        """
        model_size = model_size or settings.WHISPER_MODEL_SIZE
        logger.info(f"Iniciando transcrição de {audio_path.name} com modelo {model_size} (idioma={language})...")
        model = cls.get_model(model_size=model_size)

        # Prompt contextual: pontuação PT-BR + nomes dos participantes e glossário informados
        initial_prompt = build_initial_prompt(language, prompt_terms)

        def _run_transcribe(m):
            segments, info = m.transcribe(
                str(audio_path),
                language=language if language != "auto" else None,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=500,
                    speech_pad_ms=250
                ),
                word_timestamps=True
            )
            logger.info(f"Transcrição detectou idioma: {info.language} (probabilidade={info.language_probability:.2f})")
            total_duration = getattr(info, "duration", 0.0) or 0.0
            transcribed_segments = []
            for s in segments:
                text = s.text.strip()
                if text:
                    words_list = []
                    if hasattr(s, "words") and s.words:
                        words_list = [
                            {
                                "word": w.word.strip(),
                                "start": round(w.start, 2),
                                "end": round(w.end, 2),
                                "prob": round(getattr(w, "probability", 1.0), 2)
                            }
                            for w in s.words if w.word.strip()
                        ]
                    transcribed_segments.append({
                        "start": round(s.start, 2),
                        "end": round(s.end, 2),
                        "text": text,
                        "words": words_list
                    })
                if progress_callback and total_duration > 0:
                    progress_callback(s.end, total_duration)
            return transcribed_segments

        try:
            return _run_transcribe(model)
        except Exception as e_run:
            if "libcublas" in str(e_run).lower() or "cuda" in str(e_run).lower():
                logger.warning(f"Erro de runtime CUDA no Faster-Whisper ({e_run}). Executando transcrição via CPU (int8)...")
                cpu_model = cls.get_model(model_size=model_size, device="cpu", compute_type="int8")
                return _run_transcribe(cpu_model)
            raise e_run
