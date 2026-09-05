import sqlite3
import json
from typing import List, Optional, Dict
from datetime import datetime
from backend.config import settings
from backend.models.schemas import MeetingDetail, MeetingListItem, SpeakerSegment, MeetingMinutes


def get_connection():
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
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
    conn.commit()
    conn.close()


def save_meeting(meeting: MeetingDetail):
    conn = get_connection()
    cursor = conn.cursor()

    segments_json = json.dumps([s.model_dump() for s in meeting.segments], ensure_ascii=False)
    summary_json = json.dumps(meeting.summary.model_dump(), ensure_ascii=False) if meeting.summary else None
    speaker_map_json = json.dumps(meeting.speaker_map, ensure_ascii=False)

    cursor.execute("""
        INSERT OR REPLACE INTO meetings (
            id, title, created_at, audio_filename, audio_duration,
            segments_json, summary_json, speaker_map_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meeting.id,
        meeting.title,
        meeting.created_at,
        meeting.audio_filename,
        meeting.audio_duration,
        segments_json,
        summary_json,
        speaker_map_json
    ))
    conn.commit()
    conn.close()


def get_meeting(meeting_id: str) -> Optional[MeetingDetail]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    segments_data = json.loads(row["segments_json"])
    segments = [SpeakerSegment(**s) for s in segments_data]

    summary = None
    if row["summary_json"]:
        summary_data = json.loads(row["summary_json"])
        summary = MeetingMinutes(**summary_data)

    speaker_map = json.loads(row["speaker_map_json"]) if row["speaker_map_json"] else {}

    return MeetingDetail(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        audio_filename=row["audio_filename"],
        audio_duration=row["audio_duration"],
        audio_url=f"/api/audio/{row['audio_filename']}",
        segments=segments,
        summary=summary,
        speaker_map=speaker_map
    )


def list_meetings() -> List[MeetingListItem]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at, audio_duration, segments_json, summary_json FROM meetings ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    items = []
    for row in rows:
        segments_data = json.loads(row["segments_json"])
        speakers = set(s.get("speaker") for s in segments_data if s.get("speaker"))
        items.append(MeetingListItem(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            audio_duration=row["audio_duration"],
            speaker_count=len(speakers),
            has_summary=bool(row["summary_json"])
        ))
    return items


def delete_meeting(meeting_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# Inicializa o banco ao carregar
init_db()
