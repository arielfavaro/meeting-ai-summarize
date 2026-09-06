import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORT_DIR = DATA_DIR / "exports"
MODELS_CACHE_DIR = Path(os.getenv("HF_HOME", DATA_DIR / "models_cache"))

# Garantir criação dos diretórios necessários
for directory in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, EXPORT_DIR, MODELS_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    # Aplicação
    APP_NAME: str = "Meeting AI Summarizer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Armazenamento
    ROOT_DIR: Path = ROOT_DIR
    FRONTEND_DIR: Path = ROOT_DIR / "frontend"
    DATA_DIR: Path = DATA_DIR
    UPLOAD_DIR: Path = UPLOAD_DIR
    PROCESSED_DIR: Path = PROCESSED_DIR
    EXPORT_DIR: Path = EXPORT_DIR
    MODELS_CACHE_DIR: Path = MODELS_CACHE_DIR
    DB_PATH: Path = DATA_DIR / "meetings.db"

    # Ollama (LLM Local)
    # No Docker, pode ser "http://ollama:11434" ou "http://host.docker.internal:11434" se rodar no host
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:12b")

    # Whisper (Transcrição)
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "medium")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" ou "cuda"
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # "int8", "float16", "float32"
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "pt")

    # Diarização 100% Local e Offline (Separação de locutores nativa)
    # VAD adaptativo + Biometria acústica (MFCCs, Spectral Contrast) + Agrupamento aglomerativo
    LOCAL_DIARIZATION: bool = True
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    ENABLE_PYANNOTE: bool = os.getenv("ENABLE_PYANNOTE", "false").lower() in ("true", "1", "yes")

    # Limites
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

    class Config:
        env_file = ROOT_DIR / ".env"
        extra = "ignore"


settings = Settings()
