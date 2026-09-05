import math
import wave
import struct
from pathlib import Path


def generate_synthetic_audio(output_path: Path, duration_seconds: int = 6):
    """
    Gera um arquivo WAV mono de 16kHz com dois tons alternados simulando duas vozes.
    Voz 1: 300Hz (0s a 3s)
    Voz 2: 600Hz (3s a 6s)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    n_samples = int(duration_seconds * sample_rate)

    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            # Alterna frequência para simular dois oradores
            freq = 300.0 if t < 3.0 else 600.0
            # Adicionar envelope suave
            amplitude = 16000 * 0.5
            value = int(amplitude * math.sin(2.0 * math.pi * freq * t))
            data = struct.pack("<h", value)
            wav_file.writeframesraw(data)

    print(f"Áudio sintético de teste gerado com sucesso em: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_test.wav"
    generate_synthetic_audio(out)
