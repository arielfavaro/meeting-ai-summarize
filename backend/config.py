import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Privacidade: nenhuma telemetria de bibliotecas de IA sai da máquina.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

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
    OLLAMA_NUM_PREDICT: int = 4096      # piso de tokens para a resposta; o valor real vem do pior caso do schema
    OLLAMA_TIMEOUT: float = 900.0
    OLLAMA_KEEP_ALIVE: str = "5m"       # quanto tempo o Ollama mantém o modelo na memória após a ata

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

    # Diarização (100% local: os modelos são lidos de diretórios em MODELS_CACHE_DIR)
    # "auto" = pyannote (se o modelo estiver baixado) → SpeechBrain → acústico
    DIARIZATION_ENGINE: str = "auto"    # "auto" | "pyannote" | "speechbrain" | "acoustic"
    PYANNOTE_MODEL_DIR: Optional[Path] = None  # padrão: MODELS_CACHE_DIR/pyannote-speaker-diarization-community-1
    HF_TOKEN: str = ""                  # usado SOMENTE pelo script de download (uma vez); nunca em processamento
    ENABLE_PYANNOTE: bool = False       # legado: equivale a DIARIZATION_ENGINE=pyannote

    # GPU: liberar Whisper/diarização da VRAM antes da etapa seguinte (None = automático: só em CUDA)
    RELEASE_MODELS_AFTER_USE: Optional[bool] = None

    # Limites e execução
    MAX_FILE_SIZE_MB: int = 500
    MAX_CONCURRENT_JOBS: int = 1          # pipelines pesados simultâneos (GPU/CPU)
    JOB_RETENTION_SECONDS: int = 3600     # tempo que jobs finalizados ficam consultáveis

    @property
    def pyannote_model_dir(self) -> Path:
        return self.PYANNOTE_MODEL_DIR or (self.MODELS_CACHE_DIR / "pyannote-speaker-diarization-community-1")

    @property
    def speechbrain_model_dir(self) -> Path:
        return self.MODELS_CACHE_DIR / "speechbrain_ecapa"

    @property
    def artifacts_dir(self) -> Path:
        return self.PROCESSED_DIR / "artifacts"

    @property
    def release_models_after_use(self) -> bool:
        if self.RELEASE_MODELS_AFTER_USE is not None:
            return self.RELEASE_MODELS_AFTER_USE
        return self.WHISPER_DEVICE == "cuda"

    def ensure_dirs(self) -> None:
        for directory in (self.DATA_DIR, self.UPLOAD_DIR, self.PROCESSED_DIR, self.EXPORT_DIR, self.MODELS_CACHE_DIR):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
