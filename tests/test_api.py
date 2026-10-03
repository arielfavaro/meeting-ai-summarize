"""Testes de API: segurança de arquivos, renomeação por ID estável, datas e compatibilidade com dados legados."""
import _env  # noqa: F401

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.database import MeetingRepository
from backend.dependencies import get_file_store, get_minutes_generator, get_repo
from backend.main import app
from backend.models.schemas import MeetingDetail, MeetingMinutes, ActionItem, Decision
from backend.services.file_store import FileStore
from backend.services.minutes.generator import MinutesConfig, MinutesGenerator
from backend.services.speakers import present_meeting, resolve_labels
from fakes import FakeLLM, minutes_payload, sample_segments


def build_meeting(meeting_id="m1") -> MeetingDetail:
    segments = sample_segments()
    ids = list(dict.fromkeys(s.speaker_id for s in segments))
    return MeetingDetail(
        id=meeting_id, title="Deploy", created_at="2026-10-02T10:00:00-03:00",
        audio_filename="x_16k.wav", audio_duration=600, audio_url="/api/audio/x_16k.wav",
        segments=segments,
        summary=MeetingMinutes(
            title="Deploy", date="2026-10-02T10:00:00-03:00", participants=ids,
            executive_summary="Locutor 1 conduziu; Locutor 3 assume os testes.",
            decisions=[Decision(description="Deploy na segunda", evidence=[4])],
            action_items=[ActionItem(task="Validar testes de carga", owner="Locutor 3", evidence=[3])],
        ),
        speaker_map={i: i for i in ids}, speaker_name_sources={i: "default" for i in ids},
    )


class APITestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="meetingai-api-"))
        (self.tmp / "uploads").mkdir()
        (self.tmp / "processed").mkdir()
        self.repo = MeetingRepository(self.tmp / "test.db")
        self.files = FileStore(self.tmp / "uploads", self.tmp / "processed", max_bytes=1024)
        self.llm = FakeLLM()
        app.dependency_overrides[get_repo] = lambda: self.repo
        app.dependency_overrides[get_file_store] = lambda: self.files
        app.dependency_overrides[get_minutes_generator] = lambda: MinutesGenerator(
            self.llm, MinutesConfig(default_model="fake"))
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()


class TestFileSecurity(APITestCase):
    def test_process_rejects_path_traversal(self):
        r = self.client.post("/api/process", data={"file_id": "../../etc/passwd"})
        self.assertEqual(r.status_code, 400)

    def test_process_rejects_unknown_file(self):
        r = self.client.post("/api/process", data={"file_id": "123e4567-e89b-42d3-a456-426614174000.wav"})
        self.assertEqual(r.status_code, 404)

    def test_process_rejects_invalid_meeting_type(self):
        (self.tmp / "uploads" / "123e4567-e89b-42d3-a456-426614174000.wav").write_bytes(b"x")
        r = self.client.post("/api/process", data={
            "file_id": "123e4567-e89b-42d3-a456-426614174000.wav", "meeting_type": "festa"})
        self.assertEqual(r.status_code, 422)

    def test_audio_rejects_traversal(self):
        self.assertEqual(self.client.get("/api/audio/..%2F..%2Fsecret.db").status_code, 404)
        self.assertEqual(self.client.get("/api/audio/.env").status_code, 400)

    def test_upload_rejects_bad_extension_and_size(self):
        r = self.client.post("/api/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/upload", files={"file": ("x.wav", b"0" * 4096, "audio/wav")})
        self.assertEqual(r.status_code, 413)
        self.assertEqual(list((self.tmp / "uploads").iterdir()), [])  # arquivo parcial removido


class TestSpeakers(APITestCase):
    def test_resolve_labels_does_not_touch_longer_ids(self):
        text = "Locutor 1 falou com Locutor 10."
        self.assertEqual(resolve_labels(text, {"Locutor 1": "Ana"}), "Ana falou com Locutor 10.")

    def test_rename_updates_transcript_minutes_and_exports_without_llm(self):
        self.repo.save(build_meeting())
        r = self.client.put("/api/meetings/m1/speakers", json={"speaker_map": {"Locutor 3": "Mariana"}})
        self.assertEqual(r.status_code, 200)
        body = r.json()

        seg3 = next(s for s in body["segments"] if s["id"] == 3)
        self.assertEqual((seg3["speaker_id"], seg3["speaker"]), ("Locutor 3", "Mariana"))
        self.assertEqual(body["summary"]["action_items"][0]["owner"], "Mariana")
        self.assertIn("Mariana", body["summary"]["participants"])
        self.assertIn("Locutor 3 assume", self.repo.get("m1").summary.executive_summary)  # persistido com rótulo
        self.assertIn("Mariana assume", body["summary"]["executive_summary"])
        self.assertIn("Mariana", body["summary"]["raw_markdown"])
        self.assertEqual(self.llm.calls, [])

        md = self.client.get("/api/meetings/m1/export/md").text
        self.assertIn("**Mariana**", md)
        self.assertNotIn("Locutor 3", md)

        # Segunda renomeação continua usando o ID estável
        r = self.client.put("/api/meetings/m1/speakers", json={"speaker_map": {"Locutor 3": "Mari"}})
        self.assertEqual(r.json()["summary"]["action_items"][0]["owner"], "Mari")
        self.assertEqual(self.repo.get("m1").speaker_name_sources["Locutor 3"], "user")

    def test_regenerate_applies_llm_names_but_keeps_user_names(self):
        self.repo.save(build_meeting())
        self.client.put("/api/meetings/m1/speakers", json={"speaker_map": {"Locutor 1": "Beto"}})
        self.llm.responses = [minutes_payload()]
        r = self.client.post("/api/meetings/m1/regenerate-summary", json={})
        self.assertEqual(r.status_code, 200)
        stored = self.repo.get("m1")
        self.assertEqual(stored.speaker_map["Locutor 1"], "Beto")      # do usuário: preservado
        self.assertEqual(stored.speaker_map["Locutor 3"], "Mariana")   # validado: aplicado
        self.assertEqual(stored.speaker_name_sources["Locutor 3"], "llm")

    def test_rename_unknown_speaker_returns_400(self):
        self.repo.save(build_meeting())
        r = self.client.put("/api/meetings/m1/speakers", json={"speaker_map": {"Locutor 9": "X"}})
        self.assertEqual(r.status_code, 400)


class TestLegacyData(APITestCase):
    def test_legacy_rows_are_migrated(self):
        db = self.tmp / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE meetings (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                        audio_filename TEXT NOT NULL, audio_duration REAL NOT NULL, segments_json TEXT NOT NULL,
                        summary_json TEXT, speaker_map_json TEXT)""")
        legacy_segments = [{"id": 1, "start": 0, "end": 2, "speaker": "Ariel", "text": "Oi"},
                           {"id": 2, "start": 2, "end": 4, "speaker": "Locutor 2", "text": "Olá"}]
        legacy_summary = {"title": "Antiga", "date": "05/09/2026 15:30", "executive_summary": "Resumo",
                          "decisions": ["Decisão antiga"], "open_points": ["Ponto"],
                          "action_items": [{"task": "T", "owner": "Ariel", "deadline": "A definir", "status": "Pendente"}],
                          "suggested_speakers": {"Locutor 1": "Ariel"}, "raw_markdown": "# velho"}
        rows = [("old", "Antiga", "05/09/2026 15:30"), ("new", "Nova", "10/12/2025 09:00")]
        for mid, title, created in rows:
            conn.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?,?,?)", (
                mid, title, created, "a.wav", 60.0, json.dumps(legacy_segments), json.dumps(legacy_summary),
                json.dumps({"Locutor 1": "Ariel", "Locutor 2": "Locutor 2"})))
        conn.commit()
        conn.close()

        repo = MeetingRepository(db)
        listed = repo.list()
        # Ordenação correta após conversão para ISO (antes ordenava pelo texto dd/mm)
        self.assertEqual([m.id for m in listed], ["old", "new"])
        self.assertEqual(listed[0].created_at, "2026-09-05T15:30:00")

        m = repo.get("old")
        self.assertEqual(m.segments[0].speaker_id, "Locutor 1")  # ID estável recuperado
        self.assertEqual(m.speaker_name_sources["Locutor 1"], "user")
        self.assertEqual(m.summary.decisions[0].description, "Decisão antiga")
        view = present_meeting(m)
        self.assertEqual(view.segments[0].speaker, "Ariel")
        self.assertIn("Decisão antiga", view.summary.raw_markdown)


if __name__ == "__main__":
    unittest.main()
