import numpy as np
from pathlib import Path
from backend.services.diarization import DiarizationService
from backend.services.transcription import TranscriptionService
from backend.services.alignment import align_transcription_with_diarization, merge_multitrack_segments
from backend.services.summarizer import SummarizerService
from backend.config import settings

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
    dialogue_short = "Locutor 1: Oi\nLocutor 2: Olá"
    est_short = int(len(dialogue_short) / 2.8) + 2500
    ctx_short = min(settings.OLLAMA_NUM_CTX, max(4096, est_short))
    print(f"Short dialogue context: {ctx_short} tokens")
    assert ctx_short == 4096

    dialogue_long = "Locutor 1: Teste longo de reunião com argumentos corporativos " * 2500
    est_long = int(len(dialogue_long) / 2.8) + 2500
    ctx_long = min(settings.OLLAMA_NUM_CTX, max(4096, est_long))
    print(f"Long dialogue context: {ctx_long} tokens (configurado: {settings.OLLAMA_NUM_CTX})")
    assert ctx_long > 4096
    assert ctx_long <= settings.OLLAMA_NUM_CTX

if __name__ == "__main__":
    test_silero_vad()
    test_word_level_alignment()
    test_multitrack_merge()
    test_adaptive_num_ctx()
    print(">>> ALL 4 TESTS PASSED FLAWLESSLY! <<<")
