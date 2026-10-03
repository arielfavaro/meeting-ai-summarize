"""Persistência SQLite das reuniões (repositório) com migrações versionadas via PRAGMA user_version."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from backend.config import settings
from backend.models.schemas import (
    MeetingContext, MeetingDetail, MeetingListItem, MeetingMinutes, SpeakerSegment,
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


MIGRATIONS = [_migrate_v1, _migrate_v2]


class MeetingRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._connect() as conn:
            # Sem WAL de propósito: o banco fica em bind mount do Docker Desktop (Windows),
            # onde o arquivo -shm do WAL não é confiável.
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for idx, migration in enumerate(MIGRATIONS[version:], start=version + 1):
                migration(conn)
                conn.execute(f"PRAGMA user_version = {idx}")

    # ------------------------------------------------------------------ CRUD
    def save(self, meeting: MeetingDetail) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO meetings (
                    id, title, created_at, audio_filename, audio_duration,
                    segments_json, summary_json, speaker_map_json, context_json, speaker_sources_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                meeting.id, meeting.title, meeting.created_at, meeting.audio_filename, meeting.audio_duration,
                json.dumps([s.model_dump() for s in meeting.segments], ensure_ascii=False),
                meeting.summary.model_dump_json(exclude={"raw_markdown"}) if meeting.summary else None,
                json.dumps(meeting.speaker_map, ensure_ascii=False),
                meeting.context.model_dump_json(),
                json.dumps(meeting.speaker_name_sources, ensure_ascii=False),
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


_repository: Optional[MeetingRepository] = None


def get_repository() -> MeetingRepository:
    """Instância padrão (lazy) — usada via FastAPI Depends e substituível em testes."""
    global _repository
    if _repository is None:
        settings.ensure_dirs()
        _repository = MeetingRepository(settings.DB_PATH)
    return _repository
