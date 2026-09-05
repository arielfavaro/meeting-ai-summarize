import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from backend.config import settings

logger = logging.getLogger(__name__)


class DiarizationService:
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or settings.HF_TOKEN

    def diarize(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executa separação de oradores (Diarização).
        Tenta primeiro Pyannote.audio (se houver token HF).
        Caso contrário ou se falhar, utiliza o Diarizador Fallback baseado em clustering acústico/VAD.
        """
        if self.hf_token and settings.ENABLE_PYANNOTE:
            try:
                logger.info("Iniciando Diarização com Pyannote.audio...")
                return self._diarize_pyannote(audio_path, min_speakers, max_speakers)
            except Exception as e:
                logger.warning(f"Pyannote.audio falhou ({e}). Ativando fallback acústico...")
                return self._diarize_fallback(audio_path, min_speakers, max_speakers)
        else:
            logger.info("Token HuggingFace não fornecido. Usando Diarizador Fallback (VAD + Clustering acústico)...")
            return self._diarize_fallback(audio_path, min_speakers, max_speakers)

    def _diarize_pyannote(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        import torch
        from pyannote.audio import Pipeline

        device = torch.device("cuda" if torch.cuda.is_available() and settings.WHISPER_DEVICE == "cuda" else "cpu")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )
        except (TypeError, ValueError):
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
        pipeline.to(device)

        params = {}
        if min_speakers:
            params["min_speakers"] = min_speakers
        if max_speakers:
            params["max_speakers"] = max_speakers

        diarization = pipeline(str(audio_path), **params)

        speaker_map = {}
        speaker_counter = 1
        results = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Locutor {speaker_counter}"
                speaker_counter += 1

            results.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker_map[speaker]
            })

        return results

    def _diarize_fallback(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Diarizador Fallback 100% offline:
        1. Carrega o áudio e detecta intervalos de fala (VAD por energia e silêncio).
        2. Extrai características acústicas (MFCCs / espectrais) para cada segmento de voz.
        3. Aplica agrupamento hierárquico (Agglomerative Clustering) para separar as pessoas.
        """
        try:
            import librosa
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist

            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            duration = len(y) / sr

            if duration < 1.0:
                return [{"start": 0.0, "end": duration, "speaker": "Locutor 1"}]

            # Detectar intervalos de fala baseado em energia (top_db=28)
            intervals = librosa.effects.split(y, top_db=28, frame_length=2048, hop_length=512)

            if len(intervals) == 0:
                return [{"start": 0.0, "end": round(duration, 2), "speaker": "Locutor 1"}]

            # Fundir intervalos muito próximos (< 0.5s) e descartar muito curtos (< 0.4s)
            merged_intervals = []
            min_len = int(0.4 * sr)
            merge_gap = int(0.5 * sr)

            curr_start, curr_end = intervals[0]
            for start, end in intervals[1:]:
                if start - curr_end < merge_gap:
                    curr_end = end
                else:
                    if curr_end - curr_start >= min_len:
                        merged_intervals.append((curr_start, curr_end))
                    curr_start, curr_end = start, end
            if curr_end - curr_start >= min_len:
                merged_intervals.append((curr_start, curr_end))

            if not merged_intervals:
                merged_intervals = [(0, len(y))]

            # Extrair features de cada segmento de fala
            features = []
            valid_segments = []

            for start, end in merged_intervals:
                chunk = y[start:end]
                if len(chunk) < int(0.3 * sr):
                    continue
                # MFCC (13 coeficientes) com média e desvio padrão
                mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
                feat = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
                # Normalizar vetor
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
                features.append(feat)
                valid_segments.append((start / sr, end / sr))

            if len(features) <= 1:
                return [{"start": round(s[0], 2), "end": round(s[1], 2), "speaker": "Locutor 1"} for s in valid_segments] or [{"start": 0.0, "end": round(duration, 2), "speaker": "Locutor 1"}]

            features = np.array(features)

            # Estimar número de clusters (interlocutores)
            n_samples = len(features)
            n_clusters = 2
            if max_speakers:
                n_clusters = min(max_speakers, max(1, n_samples))
            elif min_speakers:
                n_clusters = max(min_speakers, 1)
            else:
                # Heurística automática: entre 2 e 4 interlocutores dependendo da variação
                n_clusters = 2 if n_samples < 8 else (3 if n_samples < 25 else 4)

            # Agrupamento hierárquico por distância cosseno
            dists = pdist(features, metric="cosine")
            # Tratar possíveis NaNs
            dists = np.nan_to_num(dists, nan=0.0)
            Z = linkage(dists, method="average")
            labels = fcluster(Z, t=n_clusters, criterion="maxclust")

            results = []
            for (start_sec, end_sec), cluster_id in zip(valid_segments, labels):
                results.append({
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "speaker": f"Locutor {cluster_id}"
                })

            return results

        except Exception as e:
            logger.error(f"Erro no diarizador fallback: {e}")
            import soundfile as sf
            try:
                info = sf.info(str(audio_path))
                dur = info.duration
            except Exception:
                dur = 60.0
            return [{"start": 0.0, "end": round(dur, 2), "speaker": "Locutor 1"}]
