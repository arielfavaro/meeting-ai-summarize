"""Gravações multi-faixa: canal de microfone individual e eco entre faixas."""
import _env  # noqa: F401

import unittest

from backend.models.schemas import SpeakerSegment
from backend.services.alignment import merge_multitrack_segments
from backend.services.multitrack import collapse_dominant_speaker, remove_cross_track_echo


def turn(start, end, spk):
    return {"start": start, "end": end, "speaker": spk}


def seg(i, start, end, spk, text):
    return SpeakerSegment(id=i, start=start, end=end, speaker=spk, text=text)


class TestDominantSpeaker(unittest.TestCase):
    def test_mic_channel_with_noise_speakers_becomes_one(self):
        turns = [turn(0, 50, "Locutor 1"), turn(50, 52, "Locutor 2"), turn(52, 90, "Locutor 1"), turn(90, 93, "Locutor 3")]
        out, share = collapse_dominant_speaker(turns, 0.8)
        self.assertGreater(share, 0.9)
        self.assertEqual({t["speaker"] for t in out}, {"Locutor 1"})
        self.assertEqual(len(out), 4)

    def test_real_conversation_is_kept(self):
        turns = [turn(0, 30, "Locutor 1"), turn(30, 60, "Locutor 2"), turn(60, 80, "Locutor 3")]
        out, share = collapse_dominant_speaker(turns, 0.8)
        self.assertIsNone(share)
        self.assertEqual(out, turns)

    def test_disabled(self):
        turns = [turn(0, 90, "Locutor 1"), turn(90, 91, "Locutor 2")]
        self.assertIsNone(collapse_dominant_speaker(turns, 0)[1])


class TestCrossTrackEcho(unittest.TestCase):
    def test_echo_in_mic_track_is_removed(self):
        mic = [seg(1, 0.0, 4.0, "Locutor 1", "Bom dia, vamos começar a reunião de planejamento."),
               seg(2, 10.2, 13.8, "Locutor 1", "Concordo com o prazo de sexta-feira para o deploy.")]
        call = [seg(1, 10.0, 14.0, "Locutor 2", "Concordo com o prazo de sexta feira para o deploy"),
                seg(2, 20.0, 22.0, "Locutor 3", "Perfeito, então seguimos.")]
        cleaned, removed = remove_cross_track_echo([mic, call], single_speaker_tracks={0})
        self.assertEqual(removed, 1)
        self.assertEqual([s.id for s in cleaned[0]], [1])        # eco do microfone descartado
        self.assertEqual(len(cleaned[1]), 2)                    # a chamada mantém a fala do Locutor 2
        merged = merge_multitrack_segments(cleaned)
        self.assertEqual(len(merged), 3)

    def test_different_text_at_same_time_is_kept(self):
        mic = [seg(1, 10.0, 14.0, "Locutor 1", "Eu acho que precisamos revisar o orçamento antes.")]
        call = [seg(1, 10.5, 13.0, "Locutor 2", "Pode compartilhar a tela, por favor?")]
        cleaned, removed = remove_cross_track_echo([mic, call], single_speaker_tracks={0})
        self.assertEqual(removed, 0)

    def test_short_backchannels_are_ignored(self):
        mic = [seg(1, 10.0, 10.5, "Locutor 1", "Sim.")]
        call = [seg(1, 10.0, 10.6, "Locutor 2", "Sim.")]
        self.assertEqual(remove_cross_track_echo([mic, call], {0})[1], 0)


if __name__ == "__main__":
    unittest.main()
