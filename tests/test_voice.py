"""Assinaturas de voz: seleção de janelas, fusão de locutores, biblioteca de vozes e integração."""
import _env  # noqa: F401

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.database import VoiceprintStore
from backend.services.voice.clustering import cluster_voices
from backend.services.voice.embedder import SpeakerVoice, combine, embed_speakers, select_windows, unit
from backend.services.voice.library import enroll_vector, match_voices
from backend.services.voice.service import VoiceConfig, VoiceService, relabel_with_groups

DIM = 8


def voice(direction, noise=0.0, seconds=60.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros(DIM, dtype=np.float32)
    base[direction] = 1.0
    return SpeakerVoice(unit(base + noise * rng.standard_normal(DIM).astype(np.float32)), seconds)


class TestWindows(unittest.TestCase):
    def test_longest_spans_first_split_in_equal_pieces(self):
        windows = select_windows([(0, 2), (10, 30), (40, 40.5)], min_window=1.0, max_window=8.0)
        self.assertEqual(windows[0][0], 10)                       # o trecho mais longo vem primeiro
        self.assertTrue(all(1.0 <= e - s <= 8.0 for s, e in windows))
        self.assertAlmostEqual(sum(e - s for s, e in windows), 22.0)  # 20 s + 2 s; o de 0,5 s fica de fora

    def test_budget_is_respected(self):
        windows = select_windows([(i * 10, i * 10 + 9) for i in range(100)], max_total=30.0, max_windows=40)
        self.assertLessEqual(sum(e - s for s, e in windows), 30.0 + 9.0)

    def test_embed_speakers_skips_who_barely_speaks(self):
        sr = 100
        audio = np.concatenate([np.full(20 * sr, 1.0), np.full(2 * sr, 2.0)]).astype(np.float32)

        def encode(chunks):  # "voz" = amplitude do trecho
            return np.stack([np.eye(DIM)[int(round(float(c.mean())))] for c in chunks])

        voices = embed_speakers(audio, {"A": [(0, 20)], "B": [(20, 22)]}, encode, sr=sr, min_seconds=3.0)
        self.assertEqual(set(voices), {"A"})
        self.assertAlmostEqual(float(voices["A"].vector[1]), 1.0, places=5)


class TestClustering(unittest.TestCase):
    def test_merges_same_voice_and_keeps_different_ones(self):
        voices = {"a": voice(0, 0.1, seed=1), "b": voice(0, 0.1, seed=2), "c": voice(1, 0.1, seed=3)}
        result = cluster_voices(voices, threshold=0.8)
        self.assertEqual(result.groups["a"], result.groups["b"])
        self.assertNotEqual(result.groups["a"], result.groups["c"])
        self.assertEqual(len(result.decisions), 1)

    def test_cannot_link_blocks_merge(self):
        voices = {"a": voice(0, 0.05, seed=1), "b": voice(0, 0.05, seed=2)}
        result = cluster_voices(voices, threshold=0.5, cannot_link=lambda x, y: True)
        self.assertNotEqual(result.groups["a"], result.groups["b"])

    def test_voice_service_thresholds_per_track(self):
        # mesma pessoa no microfone (faixa 0) e na chamada (faixa 1): similaridade ~0.7
        mic = SpeakerVoice(unit(np.array([1, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)), 100)
        call = SpeakerVoice(unit(np.array([1, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)), 100)
        other = voice(5, seed=4)
        svc = VoiceService(VoiceConfig(merge_threshold=0.8, cross_track_threshold=0.6))
        plan = svc.plan_merges({(0, "L1"): mic, (1, "L1"): call, (1, "L2"): other}, single_speaker_tracks={0})
        self.assertEqual(plan.groups[(0, "L1")], plan.groups[(1, "L1")])
        self.assertNotEqual(plan.groups[(1, "L1")], plan.groups[(1, "L2")])
        # mesma faixa exige o limiar mais alto
        plan = svc.plan_merges({(0, "L1"): mic, (0, "L2"): call}, single_speaker_tracks=set())
        self.assertNotEqual(plan.groups[(0, "L1")], plan.groups[(0, "L2")])

    def test_two_individual_mics_are_never_merged(self):
        svc = VoiceService(VoiceConfig(cross_track_threshold=0.1))
        plan = svc.plan_merges({(0, "L1"): voice(0), (1, "L1"): voice(0)}, single_speaker_tracks={0, 1})
        self.assertNotEqual(plan.groups[(0, "L1")], plan.groups[(1, "L1")])

    def test_relabel_with_groups_uses_global_first_appearance(self):
        diar = {0: [{"start": 5.0, "end": 9.0, "speaker": "Locutor 1"}],
                1: [{"start": 0.0, "end": 4.0, "speaker": "Locutor 1"},
                    {"start": 6.0, "end": 8.0, "speaker": "Locutor 2"},
                    {"start": 9.0, "end": 9.5, "speaker": "Locutor 3"}]}  # sem voz: fica sozinho
        voices = {(0, "Locutor 1"): voice(0), (1, "Locutor 1"): voice(1), (1, "Locutor 2"): voice(0)}
        groups = {(0, "Locutor 1"): 0, (1, "Locutor 2"): 0, (1, "Locutor 1"): 1}
        relabeled, label_voices = relabel_with_groups(diar, groups, voices)
        self.assertEqual(relabeled[1][0]["speaker"], "Locutor 1")   # fala primeiro (t=0)
        self.assertEqual(relabeled[0][0]["speaker"], "Locutor 2")   # mesma pessoa nas duas faixas
        self.assertEqual(relabeled[1][1]["speaker"], "Locutor 2")
        self.assertEqual(relabeled[1][2]["speaker"], "Locutor 3")
        self.assertEqual(set(label_voices), {"Locutor 1", "Locutor 2"})
        self.assertAlmostEqual(label_voices["Locutor 2"].seconds, 120.0)


class TestLibrary(unittest.TestCase):
    def test_match_requires_threshold_margin_and_is_one_to_one(self):
        from backend.services.voice.library import Voiceprint
        ana = Voiceprint("1", "Ana", voice(0).vector)
        bia = Voiceprint("2", "Bia", voice(1).vector)
        voices = {"Locutor 1": voice(0, 0.1, seed=1), "Locutor 2": voice(0, 0.2, seed=2),
                  "Locutor 3": voice(3, seed=3)}
        matches = match_voices(voices, [ana, bia], threshold=0.6, margin=0.05)
        self.assertEqual([(m.speaker_id, m.voiceprint.name) for m in matches], [("Locutor 1", "Ana")])

    def test_ambiguous_match_is_rejected(self):
        from backend.services.voice.library import Voiceprint
        a = Voiceprint("1", "A", unit(np.array([1, 0.9, 0, 0, 0, 0, 0, 0], dtype=np.float32)))
        b = Voiceprint("2", "B", unit(np.array([0.9, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)))
        v = SpeakerVoice(unit(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)), 60)
        self.assertEqual(match_voices({"Locutor 1": v}, [a, b], threshold=0.6, margin=0.05), [])

    def test_enrollment_accumulates_with_cap(self):
        first = enroll_vector(None, voice(0, seconds=1000), "Ana", "t1")
        self.assertEqual((first.samples, first.seconds), (1, 300.0))
        second = enroll_vector(first, voice(1, seconds=100), "Ana", "t2")
        self.assertEqual((second.samples, second.id), (2, first.id))
        self.assertGreater(second.vector[0], second.vector[1])  # a reunião longa anterior pesa mais

    def test_store_roundtrip(self):
        store = VoiceprintStore(Path(tempfile.mkdtemp()) / "db.sqlite")
        vp = enroll_vector(None, voice(2), "João Silva", "t1")
        store.save(vp)
        self.assertEqual(store.get_by_name("joao  silva").id, vp.id)   # nome normalizado
        loaded = store.list()[0]
        np.testing.assert_allclose(loaded.vector, vp.vector, rtol=1e-6)
        self.assertTrue(store.delete(vp.id))
        self.assertEqual(store.list(), [])

    def test_combine_weights_by_duration(self):
        merged = combine([voice(0, seconds=90), voice(1, seconds=10)])
        self.assertEqual(merged.seconds, 100)
        self.assertGreater(merged.vector[0], merged.vector[1])


class FakeVoiceService(VoiceService):
    """Vozes predefinidas por rótulo (sem modelo neural nem áudio de verdade)."""

    def __init__(self, by_label, store=None, **cfg):
        super().__init__(VoiceConfig(**cfg), store=store, encoder=lambda chunks: None)
        self.by_label = by_label
        self.enrolled = []

    def voices_for_turns(self, audio_path, turns):
        labels = {t["speaker"] for t in turns}
        return {label: self.by_label[label] for label in labels if label in self.by_label}

    def voices_for_segments(self, audio_path, segments):
        return {s.speaker_id: self.by_label.get(s.speaker_id) for s in segments}

    def enroll(self, name, v):
        self.enrolled.append(name)
        return super().enroll(name, v)


from backend.models.schemas import MeetingContext, ProcessOptions  # noqa: E402
from fakes import sample_segments  # noqa: E402
from test_pipeline import FILE_ID, FakeTwoTrackAudio, _PipelineBase  # noqa: E402


class TestPipelineWithVoices(_PipelineBase):
    def _turns_with_split_speaker(self):
        # O diarizador separou o Locutor 2 em dois rótulos ("Locutor 2" e "Locutor 4")
        turns = [{"start": s.start, "end": s.end, "speaker": s.speaker_id} for s in sample_segments()]
        turns[-1]["speaker"] = "Locutor 4"
        return turns

    async def test_split_speaker_is_merged_and_known_voice_is_named(self):
        store = VoiceprintStore(self.tmp / "db.sqlite")
        store.save(enroll_vector(None, voice(2, seconds=200), "Mariana Souza", "t0"))
        self.pipeline.voice = FakeVoiceService(
            {"Locutor 1": voice(0), "Locutor 2": voice(1, 0.05, seed=1), "Locutor 3": voice(2, 0.1, seed=2),
             "Locutor 4": voice(1, 0.05, seed=3)}, store=store)
        self.diarizer_turns = self._turns_with_split_speaker()
        job = self.jobs.create()
        await self.pipeline.run(job.job_id, FILE_ID, "Deploy", ProcessOptions())
        done = self.jobs.get(job.job_id)
        self.assertEqual(done.status, "completed", done.error)
        meeting = self.repo.get(done.meeting_id)
        self.assertEqual(sorted(meeting.speaker_ids()), ["Locutor 1", "Locutor 2", "Locutor 3"])
        # Reconhecido pela voz: vale mais que a sugestão do LLM ("Mariana")
        self.assertEqual(meeting.speaker_map["Locutor 3"], "Mariana Souza")
        self.assertEqual(meeting.speaker_name_sources["Locutor 3"], "voice")
        self.assertIn("identificado pela voz: Mariana Souza", self.llm.calls[0]["messages"][1]["content"])
        messages = [e.message for e in done.logs]
        self.assertTrue(any("4 rótulos de locutor → 3 pessoa(s)" in m for m in messages), messages)
        self.assertTrue(any("Voz reconhecida: Locutor 3 ➔ Mariana Souza" in m for m in messages), messages)

    async def test_exact_speaker_count_disables_within_track_merge(self):
        self.pipeline.voice = FakeVoiceService({"Locutor 1": voice(0), "Locutor 2": voice(1), "Locutor 3": voice(2),
                                                "Locutor 4": voice(1)})
        self.diarizer_turns = self._turns_with_split_speaker()
        job = self.jobs.create()
        await self.pipeline.run(job.job_id, FILE_ID, "Deploy", ProcessOptions(min_speakers=4, max_speakers=4))
        meeting = self.repo.get(self.jobs.get(job.job_id).meeting_id)
        self.assertEqual(len(meeting.speaker_ids()), 4)

    async def test_participants_cap_the_number_of_speakers(self):
        from test_pipeline import EVENTS
        job = self.jobs.create()
        await self.pipeline.run(job.job_id, FILE_ID, "Deploy",
                                ProcessOptions(context=MeetingContext(participants="Ana, Bia, Caio")))
        self.assertIn(("diarize", None, 4), EVENTS)

    async def test_same_person_on_two_tracks_becomes_one_speaker(self):
        self.pipeline.audio = FakeTwoTrackAudio
        self.pipeline.multitrack_dedupe_echo = False
        self.pipeline.multitrack_dominant_share = 0
        # As duas faixas devolvem os mesmos rótulos locais; vozes iguais => mesmas pessoas
        self.pipeline.voice = FakeVoiceService({"Locutor 1": voice(0), "Locutor 2": voice(1), "Locutor 3": voice(2)})
        job = self.jobs.create()
        await self.pipeline.run(job.job_id, FILE_ID, "Multi", ProcessOptions())
        done = self.jobs.get(job.job_id)
        self.assertEqual(done.status, "completed", done.error)
        meeting = self.repo.get(done.meeting_id)
        self.assertEqual(sorted(meeting.speaker_ids()), ["Locutor 1", "Locutor 2", "Locutor 3"])  # e não 6

    async def test_without_voice_model_pipeline_works_as_before(self):
        class Unavailable(FakeVoiceService):
            def unavailable_reason(self):
                return "modelo não baixado"
        self.pipeline.voice = Unavailable({})
        self.diarizer_turns = self._turns_with_split_speaker()
        job = self.jobs.create()
        await self.pipeline.run(job.job_id, FILE_ID, "Deploy", ProcessOptions())
        done = self.jobs.get(job.job_id)
        self.assertEqual(done.status, "completed", done.error)
        self.assertEqual(len(self.repo.get(done.meeting_id).speaker_ids()), 4)
        self.assertTrue(any("Assinaturas de voz indisponíveis" in e.message for e in done.logs))


class TestVoiceAPI(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.database import MeetingRepository
        from backend.dependencies import get_file_store, get_repo, get_voice_service
        from backend.main import app
        from backend.services.file_store import FileStore
        from test_api import build_meeting

        self.app = app
        tmp = Path(tempfile.mkdtemp(prefix="meetingai-voice-api-"))
        (tmp / "uploads").mkdir()
        (tmp / "processed").mkdir()
        (tmp / "processed" / "x_16k.wav").write_bytes(b"RIFF")
        self.repo = MeetingRepository(tmp / "db.sqlite")
        self.repo.save(build_meeting())
        self.store = VoiceprintStore(tmp / "db.sqlite")
        self.voice = FakeVoiceService({"Locutor 3": voice(2)}, store=self.store)
        app.dependency_overrides[get_repo] = lambda: self.repo
        app.dependency_overrides[get_file_store] = lambda: FileStore(tmp / "uploads", tmp / "processed", 1024)
        app.dependency_overrides[get_voice_service] = lambda: self.voice
        self.client = TestClient(app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_naming_a_speaker_enrolls_the_voice_and_it_can_be_deleted(self):
        r = self.client.put("/api/meetings/m1/speakers", json={"speaker_map": {"Locutor 3": "Mariana"}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.voice.enrolled, ["Mariana"])
        listed = self.client.get("/api/voices").json()
        self.assertEqual([v["name"] for v in listed["voices"]], ["Mariana"])
        self.assertNotIn("vector", listed["voices"][0])           # o vetor nunca sai do banco
        # a interface reenvia todos os nomes: o que não mudou não é cadastrado de novo
        self.client.put("/api/meetings/m1/speakers",
                        json={"speaker_map": {"Locutor 1": "Locutor 1", "Locutor 3": "Mariana"}})
        self.assertEqual(self.voice.enrolled, ["Mariana"])
        vid = listed["voices"][0]["id"]
        self.assertEqual(self.client.delete(f"/api/voices/{vid}").status_code, 200)
        self.assertEqual(self.client.get("/api/voices").json()["voices"], [])
        self.assertEqual(self.client.delete(f"/api/voices/{vid}").status_code, 404)


class TestPyannoteOutput(unittest.TestCase):
    def test_empty_annotation_does_not_fall_through_to_diarize_output(self):
        """Annotation vazia é "falsy": não pode cair no DiarizeOutput (AttributeError: itertracks)."""
        from unittest import mock
        from backend.services.diarization.base import has_module
        if not has_module("torch"):
            self.skipTest("torch não instalado")
        import soundfile as sf
        from backend.services.diarization.pyannote_engine import PyannoteEngine

        class EmptyAnnotation:
            def __bool__(self):
                return False

            def itertracks(self, yield_label=False):
                return iter(())

        class DiarizeOutput:  # mesmo formato do pyannote 4
            exclusive_speaker_diarization = EmptyAnnotation()
            speaker_diarization = EmptyAnnotation()

        wav = Path(tempfile.mkdtemp()) / "a.wav"
        sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
        engine = PyannoteEngine(Path(tempfile.mkdtemp()))
        with mock.patch.object(engine, "_load", return_value=lambda *a, **k: DiarizeOutput()):
            self.assertEqual(engine.diarize(wav, None, None, lambda *a, **k: None), [])


if __name__ == "__main__":
    unittest.main()
