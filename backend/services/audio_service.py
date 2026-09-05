import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class AudioService:
    @staticmethod
    def get_audio_info(file_path: Path) -> Dict[str, Any]:
        """Obtém metadados de áudio (duração, canais, sample_rate) usando ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path)
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0.0))
            return {
                "duration": duration,
                "format": data.get("format", {}).get("format_name", ""),
                "bitrate": data.get("format", {}).get("bit_rate", "")
            }
        except Exception as e:
            logger.warning(f"ffprobe falhou ({e}), tentando calcular duração com soundfile/wave...")
            try:
                import soundfile as sf
                info = sf.info(str(file_path))
                return {
                    "duration": info.duration,
                    "format": info.format,
                    "samplerate": info.samplerate
                }
            except Exception as e2:
                logger.error(f"Erro ao ler áudio com soundfile: {e2}")
                return {"duration": 0.0, "format": "unknown"}

    @staticmethod
    def convert_to_wav_16k_mono(input_path: Path, output_path: Path) -> Tuple[Path, float]:
        """
        Converte qualquer áudio/vídeo para WAV PCM 16-bit 16kHz Mono.
        Este formato é o padrão ideal para modelos Whisper e Diarização.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",  # sobrescrever
            "-i", str(input_path),
            "-vn",  # ignorar vídeo
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz
            "-ac", "1",      # Mono
            str(output_path)
        ]

        logger.info(f"Executando conversão FFmpeg: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"FFmpeg subprocess falhou ou não encontrado: {e}. Tentando librosa/soundfile...")
            import librosa
            import soundfile as sf
            y, sr = librosa.load(str(input_path), sr=16000, mono=True)
            sf.write(str(output_path), y, 16000, subtype="PCM_16")

        info = AudioService.get_audio_info(output_path)
        duration = info.get("duration", 0.0)
        return output_path, duration
