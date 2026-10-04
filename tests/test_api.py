"""Testes de API: segurança de arquivos, renomeação por ID estável, datas e compatibilidade com dados legados."""
import _env  # noqa: F401

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.database import MeetingRepository
from backend.dependencies import get_artifacts, get_file_store, get_jobs, get_minutes_generator, get_pipeline, get_repo
from backend.services.artifacts import ArtifactStore
from backend.services.jobs import JobRegistry
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
        self.artifacts = ArtifactStore(self.tmp / "processed" / "artifacts")
        self.jobs = JobRegistry()
        app.dependency_overrides[get_artifacts] = lambda: self.artifacts
        app.dependency_overrides[get_jobs] = lambda: self.jobs
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


UUID = "123e4567-e89b-42d3-a456-426614174000"


class TestSpeakerEditing(APITestCase):
    def test_merge_speakers_updates_transcript_and_minutes(self):
        self.repo.save(build_meeting())
        self.client.put("/api/meetings/m1/speakers", json={"speaker_map": {"Locutor 3": "Mariana"}})
        r = self.client.post("/api/meetings/m1/speakers/merge",
                             json={"source_ids": ["Locutor 3"], "target_id": "Locutor 2"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual({s["speaker_id"] for s in body["segments"]}, {"Locutor 1", "Locutor 2"})
        # O destino sem nome herda o nome confirmado da origem; a ata troca o rótulo
        self.assertEqual(body["summary"]["action_items"][0]["owner"], "Mariana")
        stored = self.repo.get("m1")
        self.assertNotIn("Locutor 3", stored.speaker_map)
        self.assertEqual(stored.summary.action_items[0].owner, "Locutor 2")
        self.assertEqual(self.llm.calls, [])

    def test_merge_rejects_unknown_speaker(self):
        self.repo.save(build_meeting())
        r = self.client.post("/api/meetings/m1/speakers/merge", json={"source_ids": ["Locutor 9"], "target_id": "Locutor 1"})
        self.assertEqual(r.status_code, 400)

    def test_reassign_segments_to_existing_and_new_speaker(self):
        self.repo.save(build_meeting())
        r = self.client.post("/api/meetings/m1/segments/reassign", json={"segment_ids": [5], "speaker_id": "Locutor 3"})
        self.assertEqual(next(s for s in r.json()["segments"] if s["id"] == 5)["speaker_id"], "Locutor 3")

        r = self.client.post("/api/meetings/m1/segments/reassign", json={"segment_ids": [2, 5]})
        segs = {s["id"]: s["speaker_id"] for s in r.json()["segments"]}
        self.assertEqual(segs[2], "Locutor 4")  # novo locutor (dividir)
        self.assertEqual(segs[5], "Locutor 4")

        r = self.client.post("/api/meetings/m1/segments/reassign", json={"segment_ids": [999]})
        self.assertEqual(r.status_code, 400)


class TestRobustness(APITestCase):
    def test_delete_meeting_removes_audio_and_checkpoints(self):
        meeting = build_meeting()
        meeting.audio_filename = f"{UUID}_16k.wav"
        from backend.models.schemas import MeetingSource
        meeting.source = MeetingSource(file_id=f"{UUID}.mp4", tracks=[0, 1])
        self.repo.save(meeting)
        uploads, processed = self.tmp / "uploads", self.tmp / "processed"
        for path in [uploads / f"{UUID}.mp4", processed / f"{UUID}_16k.wav", processed / f"{UUID}_track_1_16k.wav",
                     uploads / "outro-arquivo.wav"]:
            path.write_bytes(b"x")
        self.artifacts.save(f"{UUID}.mp4", "transcription", {"track": 0}, [])

        r = self.client.delete("/api/meetings/m1")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["files_removed"], 3)
        self.assertEqual([p.name for p in uploads.iterdir()], ["outro-arquivo.wav"])
        self.assertEqual(list(processed.glob("*.wav")), [])
        self.assertIsNone(self.artifacts.load(f"{UUID}.mp4", "transcription", {"track": 0}))

    def test_rediarize_legacy_meeting_returns_409(self):
        self.repo.save(build_meeting())
        r = self.client.post("/api/meetings/m1/rediarize", json={"num_speakers": 2})
        self.assertEqual(r.status_code, 409)

    def test_retry_only_failed_jobs_and_reuses_request(self):
        calls = []

        class FakePipeline:
            async def run(self, job_id, file_id, title, options):
                calls.append(("process", file_id, title, options.whisper_model))

            async def run_rediarize(self, job_id, meeting_id, request):
                calls.append(("rediarize", meeting_id, request.num_speakers))

        app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
        ok = self.jobs.create(request={"type": "process", "file_id": f"{UUID}.wav", "title": "T",
                                       "options": {"whisper_model": "small"}})
        self.assertEqual(self.client.post(f"/api/jobs/{ok.job_id}/retry").status_code, 409)

        self.jobs.update(ok.job_id, status="failed", error="boom")
        r = self.client.post(f"/api/jobs/{ok.job_id}/retry")
        self.assertEqual(r.status_code, 200)
        self.assertNotEqual(r.json()["job_id"], ok.job_id)

        red = self.jobs.create(kind="rediarize", request={"type": "rediarize", "meeting_id": "m1",
                                                          "request": {"num_speakers": 3}})
        self.jobs.update(red.job_id, status="failed")
        self.client.post(f"/api/jobs/{red.job_id}/retry")
        self.assertEqual(calls, [("process", f"{UUID}.wav", "T", "small"), ("rediarize", "m1", 3)])

    def test_active_job_endpoint(self):
        self.assertIsNone(self.client.get("/api/jobs/active").json())
        job = self.jobs.create()
        self.jobs.update(job.job_id, status="transcribing")
        self.assertEqual(self.client.get("/api/jobs/active").json()["job_id"], job.job_id)

    def test_diarization_engines_endpoint_reports_missing_models(self):
        body = self.client.get("/api/diarization/engines").json()
        names = [e["name"] for e in body["engines"]]
        self.assertEqual(names, ["pyannote", "speechbrain", "acoustic"])
        self.assertFalse(body["engines"][0]["available"])


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
