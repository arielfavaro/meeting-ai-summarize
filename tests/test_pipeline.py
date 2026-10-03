"""Pipeline ponta a ponta com dublês: não bloqueia o event loop, serializa jobs e aplica nomes validados."""
import _env  # noqa: F401

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from backend.database import MeetingRepository
from backend.models.schemas import MeetingContext, ProcessOptions
from backend.services.file_store import FileStore
from backend.services.jobs import JobRegistry
from backend.services.minutes.generator import MinutesConfig, MinutesGenerator
from backend.services.pipeline import MeetingPipeline
from fakes import FakeLLM, minutes_payload, sample_segments

FILE_ID = "123e4567-e89b-42d3-a456-426614174000.wav"


class FakeAudio:
    @staticmethod
    def detect_active_audio_tracks(path, log_callback=None):
        if log_callback:
            log_callback("Faixa #0: ativa (pico=-3.0dB, média=-20.0dB)")
        return [0]

    @staticmethod
    def convert_to_wav_16k_mono(src, dst, tracks):
        dst.write_bytes(b"RIFF")
        return dst, 20.0


class FakeDiarizer:
    def diarize(self, path, min_speakers=None, max_speakers=None, log_callback=None):
        time.sleep(0.3)  # trabalho síncrono pesado
        if log_callback:
            log_callback("Diarização finalizada: 3 locutor(es).", "success")
        return [{"start": s.start, "end": s.end, "speaker": s.speaker_id} for s in sample_segments()]


class FakeTranscriber:
    calls = []

    @classmethod
    def transcribe(cls, path, model_size, language, beam_size, progress_callback, prompt_terms):
        cls.calls.append(prompt_terms)
        time.sleep(0.3)
        progress_callback(10.0, 20.0)
        return [{"start": s.start, "end": s.end, "text": s.text, "words": []} for s in sample_segments()]


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="meetingai-pipe-"))
        (tmp / "uploads").mkdir()
        (tmp / "processed").mkdir()
        (tmp / "uploads" / FILE_ID).write_bytes(b"audio")
        self.repo = MeetingRepository(tmp / "db.sqlite")
        self.jobs = JobRegistry(max_concurrent=1)
        self.llm = FakeLLM([minutes_payload(), minutes_payload()])
        self.pipeline = MeetingPipeline(
            jobs=self.jobs,
            files=FileStore(tmp / "uploads", tmp / "processed", 10_000),
            repository=self.repo,
            minutes=MinutesGenerator(self.llm, MinutesConfig(default_model="fake")),
            processed_dir=tmp / "processed",
            diarizer_factory=lambda token: FakeDiarizer(),
            transcriber=FakeTranscriber,
            audio=FakeAudio,
        )

    async def test_end_to_end_without_blocking_event_loop(self):
        options = ProcessOptions(context=MeetingContext(participants="Carlos, Mariana", glossary="rollback"))
        job = self.jobs.create()

        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while self.jobs.get(job.job_id).status not in ("completed", "failed"):
                ticks += 1
                await asyncio.sleep(0.05)

        await asyncio.gather(self.pipeline.run(job.job_id, FILE_ID, "Deploy", options), heartbeat())

        final = self.jobs.get(job.job_id)
        self.assertEqual(final.status, "completed", final.error)
        # ~0.6s de trabalho síncrono: o loop continuou respondendo durante o processamento
        self.assertGreater(ticks, 5)
        self.assertEqual(FakeTranscriber.calls[-1], ["Carlos", "Mariana", "rollback"])

        meeting = self.repo.get(final.meeting_id)
        self.assertEqual(meeting.speaker_map["Locutor 1"], "Carlos")
        self.assertEqual(meeting.speaker_map["Locutor 3"], "Mariana")
        self.assertEqual(meeting.segments[0].speaker_id, "Locutor 1")
        self.assertEqual(final.result["segments"][0]["speaker"], "Carlos")
        self.assertEqual(final.result["summary"]["action_items"][0]["owner"], "Mariana")
        self.assertEqual(meeting.context.glossary, ["rollback"])

        # Log em tempo real: mensagens dos serviços, níveis e sequência crescente
        messages = [entry.message for entry in final.logs]
        self.assertIn("Faixa #0: ativa (pico=-3.0dB, média=-20.0dB)", messages)
        self.assertTrue(any(m.startswith("Transcrição: 50%") for m in messages), messages)
        self.assertTrue(any("Nome identificado: Locutor 1 ➔ Carlos" in m for m in messages))
        self.assertTrue(any(m.startswith("Ata gerada:") for m in messages))
        self.assertEqual(final.logs[-1].level, "success")
        seqs = [entry.seq for entry in final.logs]
        self.assertEqual(seqs, sorted(seqs))

    async def test_jobs_are_serialized(self):
        options = ProcessOptions()
        j1, j2 = self.jobs.create(), self.jobs.create()
        t1 = asyncio.create_task(self.pipeline.run(j1.job_id, FILE_ID, "A", options))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(self.pipeline.run(j2.job_id, FILE_ID, "B", options))
        await asyncio.sleep(0.05)
        self.assertIn("Aguardando", self.jobs.get(j2.job_id).current_step)
        await asyncio.gather(t1, t2)
        self.assertEqual(self.jobs.get(j2.job_id).status, "completed")

    async def test_invalid_file_fails_job(self):
        job = self.jobs.create()
        await self.pipeline.run(job.job_id, "../../etc/passwd", "X", ProcessOptions())
        failed = self.jobs.get(job.job_id)
        self.assertEqual(failed.status, "failed")
        self.assertTrue(failed.error)
        self.assertEqual(failed.logs[-1].level, "error")

    def test_log_is_capped_and_sequence_keeps_growing(self):
        from backend.services.jobs import MAX_LOG_LINES
        job = self.jobs.create()
        for i in range(MAX_LOG_LINES + 20):
            self.jobs.log(job.job_id, f"linha {i}")
        logs = self.jobs.get(job.job_id).logs
        self.assertEqual(len(logs), MAX_LOG_LINES)
        self.assertEqual(logs[-1].seq, MAX_LOG_LINES + 20)
        self.assertEqual(self.jobs.get(job.job_id).current_step, f"linha {MAX_LOG_LINES + 19}")


if __name__ == "__main__":
    unittest.main()
