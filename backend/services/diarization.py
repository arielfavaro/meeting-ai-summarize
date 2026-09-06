import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from backend.config import settings

logger = logging.getLogger(__name__)


class DiarizationService:
    """
    Motor de Diarização (Separação de Oradores) 100% Local e Offline.
    
    Funciona inteiramente na máquina local sem necessidade de conexão com a internet,
    tokens de API ou contas externas (HuggingFace).
    
    Etapas do Pipeline Local:
    1. VAD Adaptativo: Detecção inteligente de trechos de fala com fusão de pausas e sub-janelamento.
    2. Biometria Acústica: Extração de vetores de timbre vocal (MFCCs 20 coeficientes, deltas,
       contraste espectral multibanda, centróide/rolloff e zero-crossing rate).
    3. Clustering Aglomerativo: Agrupamento hierárquico por distância cosseno com seleção
       automática do número ótimo de oradores via Silhouette Score ou parâmetros manuais.
    4. Suavização Temporal: Consolidação de fatias e eliminação de ruídos transitórios.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or settings.HF_TOKEN

    def diarize(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executa separação de oradores (Diarização) 100% local e offline.
        """
        # Se o usuário configurar explicitamente Pyannote e houver token válido, tenta pyannote
        if settings.ENABLE_PYANNOTE and self.hf_token:
            try:
                logger.info("Tentando diarização com Pyannote...")
                return self._diarize_pyannote(audio_path, min_speakers, max_speakers)
            except Exception as e:
                logger.warning(f"Pyannote indisponível ({e}). Executando motor de Diarização 100% Local...")
                return self._diarize_local(audio_path, min_speakers, max_speakers)

        logger.info("Executando Diarização 100% Local (VAD + Biometria Acústica + Agrupamento Hierárquico)...")
        return self._diarize_local(audio_path, min_speakers, max_speakers)

    def _diarize_local(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Diarizador nativo 100% offline de alta fidelidade:
        - Carrega áudio 16kHz mono.
        - Segmenta atividade de voz com janelamento dinâmico.
        - Extrai assinaturas vocais (MFCCs + Contraste Espectral + Dinâmica).
        - Executa clustering por similaridade de cosseno.
        """
        try:
            import librosa
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score

            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            duration = len(y) / sr

            if duration < 1.0:
                return [{"start": 0.0, "end": round(duration, 2), "speaker": "Locutor 1"}]

            # 1. Detecção e fatiamento de fala
            segments = self._detect_speech_segments(y, sr)
            if not segments:
                return [{"start": 0.0, "end": round(duration, 2), "speaker": "Locutor 1"}]

            # 2. Extração de características acústicas
            features = []
            valid_segments = []

            for start_sec, end_sec in segments:
                start_idx = int(start_sec * sr)
                end_idx = int(end_sec * sr)
                chunk = y[start_idx:end_idx]

                feat = self._extract_vocal_features(chunk, sr)
                if feat is not None:
                    features.append(feat)
                    valid_segments.append((start_sec, end_sec))

            if len(features) <= 1:
                return [
                    {"start": round(s[0], 2), "end": round(s[1], 2), "speaker": "Locutor 1"}
                    for s in (valid_segments or [(0.0, duration)])
                ]

            features_arr = np.array(features)
            n_samples = len(features_arr)

            # 3. Determinação do número de interlocutores
            k = self._determine_k(
                features=features_arr,
                n_samples=n_samples,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

            # 4. Clustering hierárquico aglomerativo
            if k <= 1:
                labels = [0] * n_samples
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=k,
                    metric="cosine",
                    linkage="average"
                )
                labels = clusterer.fit_predict(features_arr)

            # 5. Mapeamento cronológico dos locutores
            speaker_map = {}
            counter = 1
            raw_diarized = []

            for (start_sec, end_sec), cluster_id in zip(valid_segments, labels):
                if cluster_id not in speaker_map:
                    speaker_map[cluster_id] = f"Locutor {counter}"
                    counter += 1
                raw_diarized.append({
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "speaker": speaker_map[cluster_id]
                })

            # 6. Suavização temporal e fusão de fatias contíguas
            smoothed = self._smooth_segments(raw_diarized)
            logger.info(f"Diarização Local concluída: {len(speaker_map)} locutores identificados em {len(smoothed)} segmentos.")
            return smoothed

        except Exception as e:
            logger.error(f"Erro na execução da Diarização Local: {e}", exc_info=True)
            import soundfile as sf
            try:
                info = sf.info(str(audio_path))
                dur = info.duration
            except Exception:
                dur = 60.0
            return [{"start": 0.0, "end": round(dur, 2), "speaker": "Locutor 1"}]

    def _detect_speech_segments(self, y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detecta intervalos de voz com VAD por energia e particiona trechos longos."""
        import librosa

        # Limiar de decibéis adaptativo
        intervals = librosa.effects.split(y, top_db=28, frame_length=2048, hop_length=512)
        if len(intervals) == 0:
            intervals = librosa.effects.split(y, top_db=20, frame_length=2048, hop_length=512)
        if len(intervals) == 0:
            return [(0.0, len(y) / sr)]

        # Mesclar pausas curtas (< 0.5s) e descartar trechos imperceptíveis (< 0.35s)
        merge_gap = int(0.5 * sr)
        min_len = int(0.35 * sr)
        merged = []

        curr_start, curr_end = intervals[0]
        for start, end in intervals[1:]:
            if start - curr_end < merge_gap:
                curr_end = end
            else:
                if curr_end - curr_start >= min_len:
                    merged.append((curr_start, curr_end))
                curr_start, curr_end = start, end
        if curr_end - curr_start >= min_len:
            merged.append((curr_start, curr_end))

        if not merged:
            merged = [(0, len(y))]

        # Sub-janelamento para turnos longos de fala (> 3.5s)
        # Permite identificar trocas de interlocutores sem pausas longas
        window_samples = int(2.5 * sr)
        hop_samples = int(1.25 * sr)
        slices = []

        for start, end in merged:
            length = end - start
            if length <= window_samples:
                slices.append((start / sr, end / sr))
            else:
                sub_start = start
                while sub_start + min_len < end:
                    sub_end = min(sub_start + window_samples, end)
                    slices.append((sub_start / sr, sub_end / sr))
                    if sub_end == end:
                        break
                    sub_start += hop_samples

        return slices

    def _extract_vocal_features(self, chunk: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extrai vetor acústico multimodal para identificar a assinatura do orador."""
        import librosa

        if len(chunk) < int(0.25 * sr):
            return None
        try:
            # 1. MFCC (20 coeficientes cepstrais + delta)
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            mfcc_delta = librosa.feature.delta(mfcc)
            delta_mean = np.mean(mfcc_delta, axis=1)

            # 2. Contraste Espectral (6 bandas) - captura ressonâncias e formantes vocais
            contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr, n_bands=6)
            contrast_mean = np.mean(contrast, axis=1)

            # 3. Centróide e Rolloff (brilho e frequência de corte vocal)
            centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
            spec_stats = np.array([
                np.mean(centroid), np.std(centroid),
                np.mean(rolloff), np.std(rolloff)
            ])

            # 4. Zero-Crossing Rate (taxa de cruzamento por zero)
            zcr = librosa.feature.zero_crossing_rate(chunk)
            zcr_stats = np.array([np.mean(zcr), np.std(zcr)])

            # Concatena vetor e aplica normalização L2
            feat = np.hstack([
                mfcc_mean, mfcc_std, delta_mean,
                contrast_mean, spec_stats, zcr_stats
            ])
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            norm = np.linalg.norm(feat)
            if norm > 1e-9:
                feat = feat / norm
            return feat
        except Exception:
            return None

    def _determine_k(
        self,
        features: np.ndarray,
        n_samples: int,
        min_speakers: Optional[int],
        max_speakers: Optional[int]
    ) -> int:
        """Determina o número ótimo de locutores via Silhouette Score ou limites fornecidos."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        if min_speakers and max_speakers and min_speakers == max_speakers:
            return min(min_speakers, n_samples)
        if min_speakers and not max_speakers:
            return min(min_speakers, n_samples)
        if max_speakers and not min_speakers and max_speakers == 1:
            return 1

        max_k_possible = min(6, n_samples - 1)
        if max_speakers:
            max_k_possible = min(max_speakers, max_k_possible)
        min_k_possible = max(2, min_speakers or 2)

        if max_k_possible < min_k_possible:
            return min(min_k_possible, n_samples)
        if max_k_possible == min_k_possible:
            return min_k_possible

        best_k = min_k_possible
        best_score = -2.0

        for candidate_k in range(min_k_possible, max_k_possible + 1):
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=candidate_k,
                    metric="cosine",
                    linkage="average"
                )
                pred_labels = clusterer.fit_predict(features)
                if len(set(pred_labels)) > 1:
                    score = silhouette_score(features, pred_labels, metric="cosine")
                    if score > best_score:
                        best_score = score
                        best_k = candidate_k
            except Exception:
                pass

        return best_k

    def _smooth_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Funde segmentos adjacentes do mesmo locutor para uma linha do tempo coesa."""
        if not segments:
            return []

        smoothed = []
        for seg in segments:
            if not smoothed:
                smoothed.append(dict(seg))
                continue

            last = smoothed[-1]
            if last["speaker"] == seg["speaker"] and seg["start"] - last["end"] <= 0.8:
                last["end"] = max(last["end"], seg["end"])
            else:
                smoothed.append(dict(seg))

        return smoothed

    def _diarize_pyannote(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Método opcional alternativo para quem tiver pipeline Pyannote configurado."""
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
