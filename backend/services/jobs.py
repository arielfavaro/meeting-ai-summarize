"""Registro de jobs em memória com expiração, log em tempo real e limite de concorrência dos pipelines."""
import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional

from backend.models.schemas import JobLogEntry, JobStatus

FINAL_STATES = ("completed", "failed")
MAX_LOG_LINES = 150

LogCallback = Callable[..., None]  # (message, level="info")


class JobRegistry:
    def __init__(self, retention_seconds: int = 3600, max_concurrent: int = 1):
        self._jobs: Dict[str, JobStatus] = {}
        self._seq: Dict[str, int] = {}
        self._lock = threading.Lock()  # callbacks de log chegam das threads de processamento
        self.retention_seconds = retention_seconds
        # Serializa os pipelines (Whisper/diarização/LLM disputam a mesma GPU/CPU).
        self.slots = asyncio.Semaphore(max_concurrent)

    def create(self) -> JobStatus:
        self.purge_expired()
        job = JobStatus(job_id=str(uuid.uuid4()), status="queued", progress=0,
                        current_step="Na fila para processamento...", updated_at=time.time())
        self._jobs[job.job_id] = job
        self._seq[job.job_id] = 0
        return job

    def get(self, job_id: str) -> Optional[JobStatus]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def log(self, job_id: str, message: str, level: str = "info", **fields) -> None:
        """Acrescenta uma linha ao log do job e a usa como etapa atual."""
        job = self._jobs.get(job_id)
        if not job:
            return
        with self._lock:
            self._seq[job_id] = self._seq.get(job_id, 0) + 1
            job.logs.append(JobLogEntry(seq=self._seq[job_id], time=datetime.now().strftime("%H:%M:%S"),
                                        level=level, message=message))
            if len(job.logs) > MAX_LOG_LINES:
                del job.logs[: len(job.logs) - MAX_LOG_LINES]
            job.current_step = message
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def purge_expired(self) -> None:
        now = time.time()
        expired = [jid for jid, j in self._jobs.items()
                   if j.status in FINAL_STATES and now - j.updated_at > self.retention_seconds]
        for jid in expired:
            self._jobs.pop(jid, None)
            self._seq.pop(jid, None)


class JobReporter:
    """Fachada de progresso de UM job: etapas com log, progresso silencioso e callbacks para os serviços."""

    def __init__(self, jobs: JobRegistry, job_id: str, logger: Optional[logging.Logger] = None):
        self.jobs = jobs
        self.job_id = job_id
        self.logger = logger or logging.getLogger("meeting_ai.jobs")

    def log(self, message: str, level: str = "info", **fields) -> None:
        log_fn = self.logger.error if level == "error" else self.logger.warning if level == "warning" else self.logger.info
        log_fn("[%s] %s", self.job_id, message)
        self.jobs.log(self.job_id, message, level, **fields)

    def step(self, message: str, *, status: Optional[str] = None, progress: Optional[int] = None,
             level: str = "info") -> None:
        fields = {}
        if status:
            fields["status"] = status
        if progress is not None:
            fields["progress"] = progress
        self.log(message, level, **fields)

    def progress(self, progress: int, current_step: Optional[str] = None, **fields) -> None:
        """Atualização frequente (ex.: % da transcrição) sem poluir o log."""
        if current_step is not None:
            fields["current_step"] = current_step
        self.jobs.update(self.job_id, progress=progress, **fields)

    def callback(self, prefix: str = "") -> LogCallback:
        """Callback para serviços síncronos (rodam em threads): fn(message, level='info')."""
        def _cb(message: str, level: str = "info") -> None:
            self.log(f"{prefix}{message}", level)
        return _cb
