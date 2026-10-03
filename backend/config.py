import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))


class Settings(BaseSettings):
    """
    Configuração centralizada. Toda variável pode ser sobrescrita pelo ambiente
    ou pelo arquivo .env (o pydantic-settings faz a leitura automaticamente).
    """
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # Aplicação
    APP_NAME: str = "Meeting AI Summarizer"
    APP_VERSION: str = "1.1.0"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["*"]

    # Armazenamento
    ROOT_DIR: Path = ROOT_DIR
    FRONTEND_DIR: Path = ROOT_DIR / "frontend"
    DATA_DIR: Path = DATA_DIR
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    EXPORT_DIR: Path = DATA_DIR / "exports"
    MODELS_CACHE_DIR: Path = Path(os.getenv("HF_HOME", DATA_DIR / "models_cache"))
    DB_PATH: Path = DATA_DIR / "meetings.db"

    # Ollama (LLM local)
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "gemma4:12b"
    OLLAMA_NUM_CTX: int = 32768         # teto da janela de contexto
    OLLAMA_NUM_PREDICT: int = 4096      # tokens reservados para a resposta (ata em JSON)
    OLLAMA_TIMEOUT: float = 900.0

    # Geração da ata
    MINUTES_CHUNK_TOKENS: int = 6000    # tamanho de cada bloco no modo map-reduce
    MINUTES_TEMPERATURE: float = 0.1
    MINUTES_MAX_RETRIES: int = 2
    SPEAKER_NAME_MIN_CONFIDENCE: float = 0.75

    # Whisper (transcrição)
    WHISPER_MODEL_SIZE: str = "medium"
    WHISPER_DEVICE: str = "cpu"           # "cpu" ou "cuda"
    WHISPER_COMPUTE_TYPE: str = "int8"    # "int8", "float16", "float32"
    DEFAULT_LANGUAGE: str = "pt"

    # Diarização
    HF_TOKEN: str = ""
    ENABLE_PYANNOTE: bool = False

    # Limites e execução
    MAX_FILE_SIZE_MB: int = 500
    MAX_CONCURRENT_JOBS: int = 1          # pipelines pesados simultâneos (GPU/CPU)
    JOB_RETENTION_SECONDS: int = 3600     # tempo que jobs finalizados ficam consultáveis

    def ensure_dirs(self) -> None:
        for directory in (self.DATA_DIR, self.UPLOAD_DIR, self.PROCESSED_DIR, self.EXPORT_DIR, self.MODELS_CACHE_DIR):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
