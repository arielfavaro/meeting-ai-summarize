import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Callable

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
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Erro ao ler metadados do áudio: {res.stderr}")

        data = json.loads(res.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
        channels = audio_streams[0].get("channels", 1) if audio_streams else 1
        sample_rate = int(audio_streams[0].get("sample_rate", 16000)) if audio_streams else 16000

        return {
            "duration": duration,
            "channels": channels,
            "sample_rate": sample_rate,
            "format": data.get("format", {}).get("format_name", "unknown")
        }

    @staticmethod
    def get_audio_streams_count(file_path: Path) -> int:
        """Verifica a quantidade de faixas (streams) de áudio presentes no arquivo."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a",
            str(file_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return 1
        try:
            data = json.loads(res.stdout)
            streams = data.get("streams", [])
            return max(1, len(streams))
        except Exception:
            return 1

    @staticmethod
    def detect_active_audio_tracks(file_path: Path, log_callback: Optional[Callable[[str], None]] = None) -> List[int]:
        """
        Analisa o volume de cada faixa de áudio presente no container e retorna os índices
        que contêm sinal de áudio audível (descartando canais mudos/inativos).
        """
        total_streams = AudioService.get_audio_streams_count(file_path)
        if total_streams <= 1:
            return [0]

        if log_callback:
            log_callback(f"Analisando {total_streams} faixas de áudio com filtro volumedetect do FFmpeg...")

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

                is_active = (max_vol > -50.0 or mean_vol > -70.0)
                status_desc = f"ativa (pico={max_vol:.1f}dB, média={mean_vol:.1f}dB)" if is_active else f"muda/inativa (pico={max_vol:.1f}dB)"
                logger.info(f"Faixa de áudio {i}: {status_desc}")
                if log_callback:
                    log_callback(f"Faixa #{i}: {status_desc}")

                # Se o pico for superior a -50dB ou média superior a -70dB, a faixa possui som perceptível
                if is_active:
                    active_tracks.append(i)
            except Exception as e:
                logger.warning(f"Erro ao verificar volume da faixa {i}: {e}. Considerando ativa por precaução.")
                active_tracks.append(i)

        if not active_tracks:
            logger.warning("Nenhuma faixa ativa detectada acima do limiar. Utilizando faixa 0 como padrão.")
            if log_callback:
                log_callback("Nenhuma faixa com volume acima do limiar. Usando faixa 0 como padrão.")
            return [0]

        logger.info(f"Faixas de áudio ativas detectadas ({len(active_tracks)}/{total_streams}): {active_tracks}")
        if log_callback:
            log_callback(f"Faixas ativas confirmadas: {active_tracks} ({len(active_tracks)}/{total_streams} ativas)")
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
