"""
Composition root: cria e entrega as dependências concretas para as rotas (FastAPI Depends).
Em testes, use `app.dependency_overrides[get_x] = lambda: fake`.
"""
from functools import lru_cache

from backend.config import settings
from backend.database import JobStore, MeetingRepository, VoiceprintStore, get_repository
from backend.services.artifacts import ArtifactStore
from backend.services.file_store import FileStore
from backend.services.jobs import JobRegistry
from backend.services.llm.ollama_client import OllamaClient
from backend.services.minutes.generator import MinutesConfig, MinutesGenerator
from backend.services.pipeline import MeetingPipeline
from backend.services.voice.service import VoiceConfig, VoiceService


@lru_cache(maxsize=1)
def get_jobs() -> JobRegistry:
    settings.ensure_dirs()
    return JobRegistry(retention_seconds=settings.JOB_RETENTION_SECONDS, max_concurrent=settings.MAX_CONCURRENT_JOBS,
                       store=JobStore(settings.DB_PATH))


@lru_cache(maxsize=1)
def get_file_store() -> FileStore:
    settings.ensure_dirs()
    return FileStore(settings.UPLOAD_DIR, settings.PROCESSED_DIR, (settings.MAX_FILE_SIZE_MB * 1024 * 1024) or None)


@lru_cache(maxsize=1)
def get_llm_client() -> OllamaClient:
    """Ollama (padrão) ou llama-server (LLM_PROVIDER=llamacpp); ambos com a mesma interface."""
    if settings.LLM_PROVIDER.lower() == "llamacpp":
        from backend.services.llm.llamacpp_client import LlamaCppClient
        return LlamaCppClient(settings.LLAMACPP_BASE_URL, timeout=settings.OLLAMA_TIMEOUT)
    return OllamaClient(settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_TIMEOUT, keep_alive=settings.OLLAMA_KEEP_ALIVE)


def get_minutes_generator() -> MinutesGenerator:
    return MinutesGenerator(get_llm_client(), MinutesConfig.from_settings(settings))


def get_repo() -> MeetingRepository:
    return get_repository()


@lru_cache(maxsize=1)
def get_artifacts() -> ArtifactStore:
    return ArtifactStore(settings.artifacts_dir)


@lru_cache(maxsize=1)
def get_voice_service() -> VoiceService:
    settings.ensure_dirs()
    return VoiceService(VoiceConfig.from_settings(settings), store=VoiceprintStore(settings.DB_PATH))


@lru_cache(maxsize=1)
def get_aligner():
    if settings.ALIGNMENT_ENGINE.lower() != "wav2vec2":
        return None
    from backend.services.forced_alignment import ForcedAligner
    return ForcedAligner(settings.alignment_model_dir, device=settings.WHISPER_DEVICE)


def get_pipeline() -> MeetingPipeline:
    return MeetingPipeline(
        jobs=get_jobs(),
        files=get_file_store(),
        repository=get_repo(),
        minutes=get_minutes_generator(),
        processed_dir=settings.PROCESSED_DIR,
        artifacts=get_artifacts(),
        release_models=settings.release_models_after_use,
        multitrack_dominant_share=settings.MULTITRACK_DOMINANT_SHARE,
        multitrack_dedupe_echo=settings.MULTITRACK_DEDUPE_ECHO,
        voice=get_voice_service(),
        aligner=get_aligner(),
    )
