import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

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
    def detect_active_audio_tracks(file_path: Path) -> list:
        """
        Inspeciona todas as faixas de áudio do arquivo e retorna os índices daquelas
        que contêm sinal de áudio audível (descartando canais mudos/inativos).
        """
        total_streams = AudioService.get_audio_streams_count(file_path)
        if total_streams <= 1:
            return [0]

        active_tracks = []
        for i in range(total_streams):
            try:
                cmd = [
                    "ffmpeg", "-vn",
                    "-i", str(file_path),
                    "-map", f"0:a:{i}",
                    "-af", "volumedetect",
                    "-f", "null", "-"
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                max_vol = -999.0
                mean_vol = -999.0
                for line in res.stderr.splitlines():
                    if "max_volume:" in line:
                        parts = line.split("max_volume:")[-1].strip().split()
                        if parts:
                            max_vol = float(parts[0])
                    elif "mean_volume:" in line:
                        parts = line.split("mean_volume:")[-1].strip().split()
                        if parts:
                            mean_vol = float(parts[0])

                logger.info(f"Faixa de áudio {i}: max_volume={max_vol}dB, mean_volume={mean_vol}dB")
                # Se o pico for superior a -50dB ou média superior a -70dB, a faixa possui som perceptível
                if max_vol > -50.0 or mean_vol > -70.0:
                    active_tracks.append(i)
            except Exception as e:
                logger.warning(f"Erro ao verificar volume da faixa {i}: {e}. Considerando ativa por precaução.")
                active_tracks.append(i)

        if not active_tracks:
            logger.warning("Nenhuma faixa ativa detectada acima do limiar. Utilizando faixa 0 como padrão.")
            return [0]

        logger.info(f"Faixas de áudio ativas detectadas ({len(active_tracks)}/{total_streams}): {active_tracks}")
        return active_tracks

    @staticmethod
    def extract_track_to_wav_16k(input_path: Path, track_index: int, output_path: Path) -> Tuple[Path, float]:
        """Extrai uma faixa de áudio específica para WAV PCM 16kHz mono."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-map", f"0:a:{track_index}",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path)
        ]
        logger.info(f"Extraindo faixa de áudio {track_index}: {' '.join(cmd)}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        info = AudioService.get_audio_info(output_path)
        return output_path, info.get("duration", 0.0)

    @staticmethod
    def convert_to_wav_16k_mono(input_path: Path, output_path: Path, active_tracks: Optional[list] = None) -> Tuple[Path, float]:
        """
        Converte o áudio/vídeo para WAV PCM 16-bit 16kHz Mono.
        Se houver múltiplas faixas ativas, combina-as com o filtro amix
        garantindo volume normalizado e sem faixas mudas.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if active_tracks is None:
            active_tracks = AudioService.detect_active_audio_tracks(input_path)

        num_active = len(active_tracks)
        logger.info(f"Arquivo {input_path.name}: processando {num_active} faixa(s) ativa(s): {active_tracks}")

        if num_active > 1:
            stream_inputs = "".join(f"[0:a:{i}]" for i in active_tracks)
            filter_amix = f"{stream_inputs}amix=inputs={num_active}:normalize=0[aout]"
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
            track_idx = active_tracks[0] if active_tracks else 0
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-map", f"0:a:{track_idx}",
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
