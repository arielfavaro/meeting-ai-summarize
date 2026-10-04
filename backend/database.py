"""Persistência SQLite (reuniões e jobs) com migrações versionadas via PRAGMA user_version."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional
import time

from backend.config import settings
from backend.models.schemas import (
    JobStatus, MeetingContext, MeetingDetail, MeetingListItem, MeetingMinutes, MeetingSource, SpeakerSegment,
)
from backend.services.speakers import normalize_legacy_meeting


def _migrate_v1(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            audio_filename TEXT NOT NULL,
            audio_duration REAL NOT NULL,
            segments_json TEXT NOT NULL,
            summary_json TEXT,
            speaker_map_json TEXT
        )
    """)


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """Contexto da reunião, origem dos nomes de locutores e datas em ISO 8601."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(meetings)")}
    if "context_json" not in cols:
        conn.execute("ALTER TABLE meetings ADD COLUMN context_json TEXT")
    if "speaker_sources_json" not in cols:
        conn.execute("ALTER TABLE meetings ADD COLUMN speaker_sources_json TEXT")
    for row_id, created_at in conn.execute("SELECT id, created_at FROM meetings").fetchall():
        try:
            iso = datetime.strptime(created_at, "%d/%m/%Y %H:%M").isoformat(timespec="seconds")
        except (TypeError, ValueError):
            continue  # já está em ISO
        conn.execute("UPDATE meetings SET created_at = ? WHERE id = ?", (iso, row_id))
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at)")


def _migrate_v3(conn: sqlite3.Connection) -> None:
    """Origem da reunião (re-diarização) e jobs persistidos (sobrevivem a reinícios)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(meetings)")}
    if "source_json" not in cols:
        conn.execute("ALTER TABLE meetings ADD COLUMN source_json TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            current_step TEXT,
            error TEXT,
            meeting_id TEXT,
            logs_json TEXT,
            request_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")


MIGRATIONS = [_migrate_v1, _migrate_v2, _migrate_v3]


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


_MIGRATED: set = set()


def ensure_schema(db_path: Path) -> None:
    db_path = Path(db_path)
    key = str(db_path.resolve())
    if key in _MIGRATED and db_path.exists():
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        # Sem WAL de propósito: o banco fica em bind mount do Docker Desktop (Windows),
        # onde o arquivo -shm do WAL não é confiável.
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for idx, migration in enumerate(MIGRATIONS[version:], start=version + 1):
            migration(conn)
            conn.execute(f"PRAGMA user_version = {idx}")
    _MIGRATED.add(key)


class MeetingRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        ensure_schema(self.db_path)

    def _connect(self):
        return connect(self.db_path)

    # ------------------------------------------------------------------ CRUD
    def save(self, meeting: MeetingDetail) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO meetings (
                    id, title, created_at, audio_filename, audio_duration,
                    segments_json, summary_json, speaker_map_json, context_json, speaker_sources_json, source_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                meeting.id, meeting.title, meeting.created_at, meeting.audio_filename, meeting.audio_duration,
                json.dumps([s.model_dump() for s in meeting.segments], ensure_ascii=False),
                meeting.summary.model_dump_json(exclude={"raw_markdown"}) if meeting.summary else None,
                json.dumps(meeting.speaker_map, ensure_ascii=False),
                meeting.context.model_dump_json(),
                json.dumps(meeting.speaker_name_sources, ensure_ascii=False),
                meeting.source.model_dump_json() if meeting.source else None,
            ))

    def get(self, meeting_id: str) -> Optional[MeetingDetail]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if not row:
            return None
        meeting = MeetingDetail(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            audio_filename=row["audio_filename"],
            audio_duration=row["audio_duration"],
            audio_url=f"/api/audio/{row['audio_filename']}",
            segments=[SpeakerSegment(**s) for s in json.loads(row["segments_json"])],
            summary=MeetingMinutes(**json.loads(row["summary_json"])) if row["summary_json"] else None,
            speaker_map=json.loads(row["speaker_map_json"]) if row["speaker_map_json"] else {},
            speaker_name_sources=json.loads(row["speaker_sources_json"]) if row["speaker_sources_json"] else {},
            context=MeetingContext(**json.loads(row["context_json"])) if row["context_json"] else MeetingContext(),
            source=MeetingSource(**json.loads(row["source_json"])) if row["source_json"] else None,
        )
        return normalize_legacy_meeting(meeting)

    def list(self) -> List[MeetingListItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, audio_duration, segments_json, summary_json "
                "FROM meetings ORDER BY created_at DESC"
            ).fetchall()
        items = []
        for row in rows:
            segments = json.loads(row["segments_json"])
            speakers = {s.get("speaker_id") or s.get("speaker") for s in segments}
            speakers.discard(None)
            items.append(MeetingListItem(
                id=row["id"], title=row["title"], created_at=row["created_at"],
                audio_duration=row["audio_duration"], speaker_count=len(speakers),
                has_summary=bool(row["summary_json"]),
            ))
        return items

    def delete(self, meeting_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,)).rowcount > 0


class JobStore:
    """Persistência dos jobs: status, log e parâmetros para "tentar novamente"."""

    FINAL = ("completed", "failed")

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        ensure_schema(self.db_path)

    def save(self, job: JobStatus) -> None:
        with connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs (job_id, kind, status, progress, current_step, error, meeting_id,
                                             logs_json, request_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.kind, job.status, job.progress, job.current_step, job.error, job.meeting_id,
                json.dumps([entry.model_dump() for entry in job.logs], ensure_ascii=False),
                json.dumps(job.request, ensure_ascii=False) if job.request is not None else None,
                job.created_at or time.time(), job.updated_at or time.time(),
            ))

    def get(self, job_id: str) -> Optional[JobStatus]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._from_row(row) if row else None

    def latest_active(self) -> Optional[JobStatus]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE status NOT IN ('completed', 'failed') "
                               "ORDER BY updated_at DESC LIMIT 1").fetchone()
        return self._from_row(row) if row else None

    def mark_interrupted(self, message: str) -> int:
        """Na subida do servidor: jobs que estavam rodando foram interrompidos pelo reinício."""
        with connect(self.db_path) as conn:
            return conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, current_step = ?, updated_at = ? "
                "WHERE status NOT IN ('completed', 'failed')", (message, message, time.time())).rowcount

    def purge_older_than(self, seconds: float) -> None:
        with connect(self.db_path) as conn:
            conn.execute("DELETE FROM jobs WHERE status IN ('completed', 'failed') AND updated_at < ?",
                         (time.time() - seconds,))

    @staticmethod
    def _from_row(row) -> JobStatus:
        return JobStatus(
            job_id=row["job_id"], kind=row["kind"], status=row["status"], progress=row["progress"],
            current_step=row["current_step"] or "", error=row["error"], meeting_id=row["meeting_id"],
            logs=json.loads(row["logs_json"]) if row["logs_json"] else [],
            request=json.loads(row["request_json"]) if row["request_json"] else None,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )


_repository: Optional[MeetingRepository] = None


def get_repository() -> MeetingRepository:
    """Instância padrão (lazy) — usada via FastAPI Depends e substituível em testes."""
    global _repository
    if _repository is None:
        settings.ensure_dirs()
        _repository = MeetingRepository(settings.DB_PATH)
    return _repository
