"""Alinhamento forçado (CTC/Viterbi em numpy) e integração no pipeline."""
import _env  # noqa: F401

import unittest
from pathlib import Path

import numpy as np

from backend.services.forced_alignment import ForcedAligner, build_tokens, ctc_align, refine_segment_words

VOCAB = {"<pad>": 0, "|": 1, "a": 2, "b": 3, "c": 4, "o": 5, "ã": 6}


def emissions(frame_tokens, vocab_size=len(VOCAB)):
    """Log-probs com o token indicado muito provável em cada quadro."""
    lp = np.full((len(frame_tokens), vocab_size), np.log(0.01 / (vocab_size - 1)))
    for t, tok in enumerate(frame_tokens):
        lp[t, tok] = np.log(0.99)
    return lp


class TestCTC(unittest.TestCase):
    def test_viterbi_finds_token_frames(self):
        # quadros: 0-3 blank | 4-6 "a" | 7 blank | 8-9 "b" | 10-11 "|" | 12-15 "c" | 16-19 blank
        frames = [0] * 4 + [2] * 3 + [0] + [3] * 2 + [1] * 2 + [4] * 4 + [0] * 4
        tokens, spans = build_tokens(["Ab,", "c!"], VOCAB, "|")
        self.assertEqual(tokens, [2, 3, 1, 4])
        result = ctc_align(emissions(frames), tokens, blank=0)
        self.assertEqual(result, [(4, 6), (8, 9), (10, 11), (12, 15)])

    def test_impossible_alignment_returns_none(self):
        self.assertIsNone(ctc_align(emissions([2, 3]), [2, 3, 4, 5], blank=0))

    def test_accents_fall_back_to_base_letter(self):
        tokens, _ = build_tokens(["Ção"], {"<pad>": 0, "c": 1, "a": 2, "o": 3}, "|")
        self.assertEqual(tokens, [1, 2, 3])

    def test_refine_moves_words_to_real_audio_and_interpolates_numbers(self):
        frames = [0] * 10 + [2] * 5 + [3] * 5 + [1] + [0] * 9 + [4] * 5 + [0] * 5   # 40 quadros
        words = [{"word": "ab", "start": 0.0, "end": 0.2}, {"word": "2026", "start": 0.2, "end": 0.5},
                 {"word": "c", "start": 0.5, "end": 0.6}]
        out = refine_segment_words(words, emissions(frames), frame_sec=0.02, offset=10.0, vocab=VOCAB,
                                   blank=0, delimiter="|")
        self.assertAlmostEqual(out[0]["start"], 10.2)        # "ab" começa no quadro 10
        self.assertAlmostEqual(out[0]["end"], 10.4)
        self.assertAlmostEqual(out[2]["start"], 10.6)        # "c" no quadro 30
        self.assertEqual((out[1]["start"], out[1]["end"]), (out[0]["end"], out[2]["start"]))  # número interpolado
        self.assertTrue(out[0]["aligned"] and not out[1].get("aligned"))


class TestAlignerService(unittest.TestCase):
    def test_refine_keeps_whisper_times_when_segment_cannot_be_aligned(self):
        import soundfile as sf
        import tempfile
        path = Path(tempfile.mkdtemp()) / "a.wav"
        sf.write(str(path), np.zeros(16000 * 3, dtype=np.float32), 16000)

        def fake_emissions(audio):  # 1 quadro a cada 20 ms, tudo "a"
            return emissions([2] * max(1, len(audio) // 320))

        aligner = ForcedAligner(Path("/nao/existe"), emissions=fake_emissions, vocab=VOCAB, blank=0, delimiter="|")
        self.assertIsNone(aligner.unavailable_reason())
        segs = [{"start": 0.5, "end": 1.5, "text": "a", "words": [{"word": "a", "start": 0.6, "end": 0.9}]},
                {"start": 1.6, "end": 1.62, "text": "bbbbbbbbbbbb",
                 "words": [{"word": "bbbbbbbbbbbb", "start": 1.6, "end": 1.62}]},
                {"start": 2.0, "end": 2.5, "text": "", "words": []}]
        out, ok, tried = aligner.refine(path, segs)
        self.assertEqual((ok, tried), (1, 2))
        self.assertTrue(out[0]["words"][0]["aligned"])
        self.assertEqual(out[1], segs[1])     # curto demais para 12 letras: fica como o Whisper deu
        self.assertEqual(out[2], segs[2])

    def test_missing_model_is_reported(self):
        reason = ForcedAligner(Path("/nao/existe")).unavailable_reason()
        self.assertIsNotNone(reason)


class FakeAligner:
    model_dir = Path("fake-aligner")

    def __init__(self):
        self.calls = 0
        self.released = False

    def unavailable_reason(self):
        return None

    def refine(self, path, segments, progress=None):
        self.calls += 1
        return [{**s, "words": [{**w, "aligned": True} for w in s.get("words", [])]} for s in segments], len(segments), len(segments)

    def release(self):
        self.released = True


from backend.models.schemas import ProcessOptions  # noqa: E402
from test_pipeline import FILE_ID, _PipelineBase  # noqa: E402


class TestPipelineAlignment(_PipelineBase):
    async def test_aligner_runs_once_and_is_cached(self):
        aligner = self.pipeline.aligner = FakeAligner()
        job = self.jobs.create()
        await self.pipeline.run(job.job_id, FILE_ID, "A", ProcessOptions())
        done = self.jobs.get(job.job_id)
        self.assertEqual(done.status, "completed", done.error)
        self.assertEqual(aligner.calls, 1)
        self.assertTrue(aligner.released)
        self.assertTrue(any("Palavras alinhadas ao áudio" in e.message for e in done.logs))
        job2 = self.jobs.create()
        await self.pipeline.run(job2.job_id, FILE_ID, "A", ProcessOptions())
        self.assertEqual(aligner.calls, 1)   # reaproveitado do checkpoint


if __name__ == "__main__":
    unittest.main()
