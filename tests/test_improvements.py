import _env  # noqa: F401  (ambiente isolado)

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.services.diarization import DiarizationService
from backend.services.alignment import align_transcription_with_diarization, merge_multitrack_segments
from backend.services.minutes.generator import MinutesConfig, MinutesGenerator
from backend.services.transcription import build_initial_prompt

HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None and importlib.util.find_spec("librosa") is not None


@unittest.skipUnless(HAS_FASTER_WHISPER, "faster-whisper/librosa não instalados")
def test_silero_vad():
    print("=== TEST 1: Silero VAD speech detection ===")
    diarizer = DiarizationService()
    # Test on synthetic audio
    dummy = np.random.randn(48000).astype(np.float32)
    segments = diarizer._detect_speech_segments(dummy, sr=16000)
    print("VAD segments detected count:", len(segments))
    assert len(segments) >= 1

def test_word_level_alignment():
    print("=== TEST 2: Alignment with word-level splitting ===")
    transcription_sample = [
        {
            "start": 1.0,
            "end": 5.0,
            "text": "Olá Ariel tudo bem com você",
            "words": [
                {"word": "Olá", "start": 1.0, "end": 1.5},
                {"word": "Ariel", "start": 1.6, "end": 2.2},
                {"word": "tudo", "start": 2.8, "end": 3.2},
                {"word": "bem", "start": 3.3, "end": 3.8},
                {"word": "com", "start": 3.9, "end": 4.2},
                {"word": "você", "start": 4.3, "end": 4.9}
            ]
        }
    ]
    diarization_sample = [
        {"start": 0.8, "end": 2.4, "speaker": "Locutor 1"},
        {"start": 2.7, "end": 5.2, "speaker": "Locutor 2"}
    ]
    aligned = align_transcription_with_diarization(transcription_sample, diarization_sample)
    print("Word-level aligned segments:")
    for a in aligned:
        print(f"  [{a.start}s - {a.end}s] {a.speaker}: '{a.text}'")
    assert len(aligned) == 2, f"Expected 2 segments due to speaker switch, got {len(aligned)}"
    assert aligned[0].speaker == "Locutor 1"
    assert aligned[1].speaker == "Locutor 2"
    assert aligned[0].text == "Olá Ariel"
    assert aligned[1].text == "tudo bem com você"

def test_multitrack_merge():
    print("=== TEST 3: Multi-track segment merging ===")
    track_1 = [
        align_transcription_with_diarization(
            [{"start": 1.0, "end": 3.0, "text": "Fala faixa 1"}],
            [{"start": 1.0, "end": 3.0, "speaker": "Locutor 1"}]
        )[0]
    ]
    track_2 = [
        align_transcription_with_diarization(
            [{"start": 2.5, "end": 5.0, "text": "Fala faixa 2"}],
            [{"start": 2.5, "end": 5.0, "speaker": "Locutor 1"}]
        )[0]
    ]
    merged = merge_multitrack_segments([track_1, track_2])
    print("Merged multi-track segments:")
    for m in merged:
        print(f"  ID {m.id} [{m.start} - {m.end}] {m.speaker}: '{m.text}'")
    assert len(merged) == 2
    assert merged[0].speaker == "Locutor 1"
    assert merged[1].speaker == "Locutor 2"

def test_adaptive_num_ctx():
    print("=== TEST 4: Adaptive num_ctx calculation ===")
    gen = MinutesGenerator(llm=None, config=MinutesConfig(default_model="x", max_ctx=32768, num_predict=4096))
    short = [{"role": "user", "content": "Locutor 1: Oi\nLocutor 2: Olá"}]
    assert gen._num_ctx_for(short, 0) == 4096           # piso mínimo
    assert 4096 <= gen._num_ctx_for(short, 4096) < 4300  # prompt curto + espaço para a resposta
    long = [{"role": "user", "content": "Locutor 1: Teste longo de reunião com argumentos corporativos " * 2500}]
    ctx_long = gen._num_ctx_for(long, 4096)
    print(f"Long dialogue context: {ctx_long} tokens")
    assert 4096 < ctx_long <= 32768


def test_diarization_falls_back_to_local_engine():
    """Antes, uma falha do SpeechBrain fazia diarize() retornar None (tudo virava 'Locutor 1')."""
    print("=== TEST 5: Diarization fallback chain ===")
    diarizer = DiarizationService()
    expected = [{"start": 0.0, "end": 1.0, "speaker": "Locutor 1"}, {"start": 1.0, "end": 2.0, "speaker": "Locutor 2"}]
    with mock.patch.object(diarizer, "_diarize_speechbrain", side_effect=RuntimeError("sem speechbrain")), \
         mock.patch.object(diarizer, "_diarize_local", return_value=expected) as local:
        result = diarizer.diarize(Path("qualquer.wav"))
    assert result == expected
    local.assert_called_once()


def test_whisper_initial_prompt_includes_context_terms():
    print("=== TEST 6: Whisper initial prompt ===")
    prompt = build_initial_prompt("pt", ["Ariel", "Mariana", "Kubernetes", "Ariel"])
    assert "Vocabulário: Ariel, Mariana, Kubernetes." in prompt
    assert len(build_initial_prompt("pt", ["termo"] * 500)) <= 600


class TestImprovements(unittest.TestCase):
    """Permite rodar via `python -m unittest discover tests`."""

    @unittest.skipUnless(HAS_FASTER_WHISPER, "faster-whisper/librosa não instalados")
    def test_silero_vad(self):
        test_silero_vad()

    def test_word_level_alignment(self):
        test_word_level_alignment()

    def test_multitrack_merge(self):
        test_multitrack_merge()

    def test_adaptive_num_ctx(self):
        test_adaptive_num_ctx()

    def test_diarization_fallback(self):
        test_diarization_falls_back_to_local_engine()

    def test_whisper_prompt(self):
        test_whisper_initial_prompt_includes_context_terms()


if __name__ == "__main__":
    unittest.main()
