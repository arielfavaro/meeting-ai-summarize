"""Registro de jobs em memória com expiração e limite de concorrência dos pipelines pesados."""
import asyncio
import time
import uuid
from typing import Dict, Optional

from backend.models.schemas import JobStatus

FINAL_STATES = ("completed", "failed")


class JobRegistry:
    def __init__(self, retention_seconds: int = 3600, max_concurrent: int = 1):
        self._jobs: Dict[str, JobStatus] = {}
        self.retention_seconds = retention_seconds
        # Serializa os pipelines (Whisper/diarização/LLM disputam a mesma GPU/CPU).
        self.slots = asyncio.Semaphore(max_concurrent)

    def create(self) -> JobStatus:
        self.purge_expired()
        job = JobStatus(job_id=str(uuid.uuid4()), status="queued", progress=0,
                        current_step="Na fila para processamento...", updated_at=time.time())
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[JobStatus]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()

    def purge_expired(self) -> None:
        now = time.time()
        expired = [jid for jid, j in self._jobs.items()
                   if j.status in FINAL_STATES and now - j.updated_at > self.retention_seconds]
        for jid in expired:
            self._jobs.pop(jid, None)
