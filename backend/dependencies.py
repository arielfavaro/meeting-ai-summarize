"""
Composition root: cria e entrega as dependências concretas para as rotas (FastAPI Depends).
Em testes, use `app.dependency_overrides[get_x] = lambda: fake`.
"""
from functools import lru_cache

from backend.config import settings
from backend.database import MeetingRepository, get_repository
from backend.services.file_store import FileStore
from backend.services.jobs import JobRegistry
from backend.services.llm.ollama_client import OllamaClient
from backend.services.minutes.generator import MinutesConfig, MinutesGenerator
from backend.services.pipeline import MeetingPipeline


@lru_cache(maxsize=1)
def get_jobs() -> JobRegistry:
    return JobRegistry(retention_seconds=settings.JOB_RETENTION_SECONDS, max_concurrent=settings.MAX_CONCURRENT_JOBS)


@lru_cache(maxsize=1)
def get_file_store() -> FileStore:
    settings.ensure_dirs()
    return FileStore(settings.UPLOAD_DIR, settings.PROCESSED_DIR, settings.MAX_FILE_SIZE_MB * 1024 * 1024)


@lru_cache(maxsize=1)
def get_llm_client() -> OllamaClient:
    return OllamaClient(settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_TIMEOUT)


def get_minutes_generator() -> MinutesGenerator:
    return MinutesGenerator(get_llm_client(), MinutesConfig.from_settings(settings))


def get_repo() -> MeetingRepository:
    return get_repository()


def get_pipeline() -> MeetingPipeline:
    return MeetingPipeline(
        jobs=get_jobs(),
        files=get_file_store(),
        repository=get_repo(),
        minutes=get_minutes_generator(),
        processed_dir=settings.PROCESSED_DIR,
    )
