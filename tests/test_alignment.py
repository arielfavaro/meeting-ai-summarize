"""Atribuição de locutor por frase (sem modelos)."""
import _env  # noqa: F401

import time
import unittest

from backend.services.alignment import SpeakerTimeline, align_transcription_with_diarization


def words(*items):
    """items: (texto, início, fim)"""
    return [{"word": t, "start": s, "end": e} for t, s, e in items]


def seg(ws):
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": " ".join(w["word"] for w in ws), "words": ws}


class TestSentenceAlignment(unittest.TestCase):
    def test_single_word_at_turn_edge_does_not_split_sentence(self):
        ws = words(("Vamos", 0.0, 0.4), ("fechar", 0.45, 0.8), ("o", 0.85, 0.9), ("deploy", 0.95, 1.4),
                   ("na", 1.45, 1.6), ("segunda.", 1.65, 2.2))
        diar = [{"start": 0.0, "end": 1.42, "speaker": "Locutor 1"},
                {"start": 1.42, "end": 1.62, "speaker": "Locutor 2"},  # erro típico na borda
                {"start": 1.62, "end": 2.3, "speaker": "Locutor 1"}]
        out = align_transcription_with_diarization([seg(ws)], diar)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].speaker, "Locutor 1")
        self.assertEqual(out[0].text, "Vamos fechar o deploy na segunda.")

    def test_punctuation_separates_speakers(self):
        ws = words(("Pode", 0.0, 0.3), ("ser.", 0.35, 0.7), ("Concordo", 0.9, 1.5), ("plenamente.", 1.55, 2.3))
        diar = [{"start": 0.0, "end": 0.8, "speaker": "Locutor 1"}, {"start": 0.8, "end": 2.4, "speaker": "Locutor 2"}]
        out = align_transcription_with_diarization([seg(ws)], diar)
        self.assertEqual([(s.speaker, s.text) for s in out],
                         [("Locutor 1", "Pode ser."), ("Locutor 2", "Concordo plenamente.")])

    def test_real_interruption_inside_sentence_is_kept(self):
        ws = words(("eu", 0.0, 0.2), ("acho", 0.25, 0.5), ("que", 0.55, 0.7), ("o", 0.75, 0.8),
                   ("prazo", 0.85, 1.2), ("mas", 1.3, 1.5), ("isso", 1.55, 1.9), ("não", 1.95, 2.3),
                   ("fecha", 2.35, 2.8), ("nunca", 2.85, 3.3))
        diar = [{"start": 0.0, "end": 1.25, "speaker": "Locutor 1"}, {"start": 1.25, "end": 3.4, "speaker": "Locutor 2"}]
        out = align_transcription_with_diarization([seg(ws)], diar)
        self.assertEqual([s.speaker for s in out], ["Locutor 1", "Locutor 2"])
        self.assertEqual(out[1].text, "mas isso não fecha nunca")

    def test_short_backchannel_from_other_speaker_is_kept(self):
        a = words(("Então", 0.0, 0.4), ("seguimos", 0.45, 1.0), ("assim.", 1.05, 1.5))
        b = words(("Sim.", 1.7, 1.95))
        c = words(("E", 2.2, 2.3), ("depois", 2.35, 2.8), ("revisamos.", 2.85, 3.5))
        diar = [{"start": 0.0, "end": 1.6, "speaker": "Locutor 1"}, {"start": 1.6, "end": 2.0, "speaker": "Locutor 2"},
                {"start": 2.0, "end": 3.6, "speaker": "Locutor 1"}]
        out = align_transcription_with_diarization([seg(a), seg(b), seg(c)], diar)
        self.assertEqual([s.speaker for s in out], ["Locutor 1", "Locutor 2", "Locutor 1"])

    def test_long_monologue_is_capped_for_precise_evidence(self):
        ws = []
        t = 0.0
        for i in range(40):  # 40 frases de ~2s do mesmo locutor
            ws += words((f"Frase{i}", t, t + 1.0), ("final.", t + 1.05, t + 1.9))
            t += 2.0
        out = align_transcription_with_diarization([seg(ws)], [{"start": 0, "end": t, "speaker": "Locutor 1"}])
        self.assertGreater(len(out), 1)
        self.assertTrue(all(s.end - s.start <= 30.0 for s in out))

    def test_segments_without_words_fall_back_to_segment_level(self):
        out = align_transcription_with_diarization(
            [{"start": 0.0, "end": 2.0, "text": "Sem palavras"}],
            [{"start": 0.0, "end": 2.0, "speaker": "Locutor 2"}])
        self.assertEqual((out[0].speaker, out[0].text), ("Locutor 2", "Sem palavras"))

    def test_timeline_is_fast_for_long_meetings(self):
        turns = [{"start": i * 2.0, "end": i * 2.0 + 2.0, "speaker": f"Locutor {i % 4 + 1}"} for i in range(3000)]
        timeline = SpeakerTimeline(turns)
        started = time.perf_counter()
        for i in range(20000):
            timeline.speaker_at(i * 0.3, i * 0.3 + 0.25)
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(timeline.speaker_at(2.1, 2.5), "Locutor 2")
        self.assertEqual(timeline.speaker_at(99999, 100000), "Locutor 4")  # sem sobreposição: mais próximo


if __name__ == "__main__":
    unittest.main()
