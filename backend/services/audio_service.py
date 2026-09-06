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
    def get_audio_streams_count(file_path: Path) -> int:
        """Detecta a quantidade de faixas de áudio no arquivo usando ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "json",
            str(file_path)
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            return len(streams)
        except Exception as e:
            logger.warning(f"ffprobe erro ao inspecionar faixas de áudio: {e}")
            return 1

    @staticmethod
    def convert_to_wav_16k_mono(input_path: Path, output_path: Path) -> Tuple[Path, float]:
        """
        Converte qualquer áudio/vídeo para WAV PCM 16-bit 16kHz Mono.
        Se o arquivo contiver múltiplas faixas de áudio (comum em gravações de OBS Studio,
        onde microfones e sons do desktop ficam em faixas separadas), combina todas as faixas
        usando o filtro amix do FFmpeg para não perder nenhuma fala.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_streams = AudioService.get_audio_streams_count(input_path)
        logger.info(f"Arquivo {input_path.name} possui {num_streams} faixa(s) de áudio.")

        if num_streams > 1:
            stream_inputs = "".join(f"[0:a:{i}]" for i in range(num_streams))
            filter_amix = f"{stream_inputs}amix=inputs={num_streams}:normalize=0[aout]"
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-filter_complex", filter_amix,
                "-map", "[aout]",
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(output_path)
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(output_path)
            ]

        logger.info(f"Executando conversão FFmpeg: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"FFmpeg com filtro falhou ({e}). Tentando conversão simples...")
            cmd_simple = [
                "ffmpeg", "-y", "-i", str(input_path), "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(output_path)
            ]
            subprocess.run(cmd_simple, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        info = AudioService.get_audio_info(output_path)
        duration = info.get("duration", 0.0)
        return output_path, duration
