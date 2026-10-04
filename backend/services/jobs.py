"""
Registro de jobs: memória (tempo real) + SQLite (sobrevive a reinícios), log em tempo real e
limite de concorrência dos pipelines.
"""
import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from backend import clock
from backend.models.schemas import JobLogEntry, JobStatus

FINAL_STATES = ("completed", "failed")
MAX_LOG_LINES = 150
PROGRESS_PERSIST_INTERVAL = 2.0  # s — progresso "silencioso" é gravado no banco no máximo a cada 2s
INTERRUPTED_MESSAGE = "Processamento interrompido pelo reinício do servidor. Use \"Tentar novamente\"."

logger = logging.getLogger(__name__)

LogCallback = Callable[..., None]  # (message, level="info")


class JobRegistry:
    def __init__(self, retention_seconds: int = 3600, max_concurrent: int = 1, store=None):
        self._jobs: Dict[str, JobStatus] = {}
        self._seq: Dict[str, int] = {}
        self._last_persist: Dict[str, float] = {}
        self._lock = threading.Lock()  # callbacks de log chegam das threads de processamento
        self.retention_seconds = retention_seconds
        self.store = store  # JobStore (SQLite) ou None (somente memória, ex.: testes)
        # Serializa os pipelines (Whisper/diarização/LLM disputam a mesma GPU/CPU).
        self.slots = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------ persistência
    def _persist(self, job: JobStatus, force: bool = False) -> None:
        if not self.store:
            return
        now = time.time()
        if not force and now - self._last_persist.get(job.job_id, 0) < PROGRESS_PERSIST_INTERVAL:
            return
        self._last_persist[job.job_id] = now
        try:
            with self._lock:
                snapshot = job.model_copy(deep=True)
            self.store.save(snapshot)
        except Exception:  # o banco nunca pode derrubar o processamento
            logger.warning("Falha ao gravar job %s no banco", job.job_id, exc_info=True)

    def recover_interrupted(self) -> int:
        """Chamado na subida: jobs que estavam rodando quando o servidor caiu viram 'failed' com retry."""
        if not self.store:
            return 0
        return self.store.mark_interrupted(INTERRUPTED_MESSAGE)

    # ------------------------------------------------------------------- API
    def create(self, kind: str = "process", request: Optional[Dict[str, Any]] = None) -> JobStatus:
        self.purge_expired()
        now = time.time()
        job = JobStatus(job_id=str(uuid.uuid4()), kind=kind, status="queued", progress=0,
                        current_step="Na fila para processamento...", request=request,
                        created_at=now, updated_at=now)
        self._jobs[job.job_id] = job
        self._seq[job.job_id] = 0
        self._persist(job, force=True)
        return job

    def get(self, job_id: str) -> Optional[JobStatus]:
        job = self._jobs.get(job_id)
        if job is None and self.store:
            try:
                job = self.store.get(job_id)
            except Exception:
                logger.warning("Falha ao ler job %s do banco", job_id, exc_info=True)
        return job

    def update(self, job_id: str, **fields) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        with self._lock:
            status_changed = "status" in fields and fields["status"] != job.status
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()
        self._persist(job, force=status_changed or "error" in fields or "meeting_id" in fields)

    def log(self, job_id: str, message: str, level: str = "info", **fields) -> None:
        """Acrescenta uma linha ao log do job e a usa como etapa atual."""
        job = self._jobs.get(job_id)
        if not job:
            return
        with self._lock:
            self._seq[job_id] = self._seq.get(job_id, 0) + 1
            stamp = clock.now()
            job.logs.append(JobLogEntry(seq=self._seq[job_id], time=stamp.strftime("%H:%M:%S"), ts=stamp.timestamp(),
                                        level=level, message=message))
            if len(job.logs) > MAX_LOG_LINES:
                del job.logs[: len(job.logs) - MAX_LOG_LINES]
            job.current_step = message
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()
        self._persist(job, force=True)

    def active(self) -> Optional[JobStatus]:
        """Job em andamento mais recente (para a interface reconectar após recarregar a página)."""
        running = [j for j in self._jobs.values() if j.status not in FINAL_STATES]
        return max(running, key=lambda j: j.updated_at) if running else None

    def purge_expired(self) -> None:
        now = time.time()
        expired = [jid for jid, j in self._jobs.items()
                   if j.status in FINAL_STATES and now - j.updated_at > self.retention_seconds]
        for jid in expired:
            self._jobs.pop(jid, None)
            self._seq.pop(jid, None)
            self._last_persist.pop(jid, None)


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
