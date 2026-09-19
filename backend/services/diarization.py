import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from backend.config import settings

logger = logging.getLogger(__name__)

_SPEECHBRAIN_CLASSIFIER = None


def get_speechbrain_classifier():
    """Carrega e mantém o modelo SpeechBrain ECAPA-TDNN em cache de memória."""
    global _SPEECHBRAIN_CLASSIFIER
    if _SPEECHBRAIN_CLASSIFIER is None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = str(settings.MODELS_CACHE_DIR / "speechbrain_ecapa")
        device = "cuda:0" if torch.cuda.is_available() and settings.WHISPER_DEVICE == "cuda" else "cpu"
        logger.info(f"Carregando classificador SpeechBrain ECAPA-TDNN no dispositivo: {device}...")
        _SPEECHBRAIN_CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": device}
        )
        logger.info("Classificador SpeechBrain carregado com sucesso!")
    return _SPEECHBRAIN_CLASSIFIER


class DiarizationService:
    """
    Motor de Diarização (Separação de Oradores) 100% Local e Offline.
    
    Funciona inteiramente na máquina local sem necessidade de conexão com a internet,
    tokens de API ou contas externas (HuggingFace).
    
    Etapas do Pipeline Local:
    1. VAD Adaptativo: Detecção inteligente de trechos de fala com fusão de pausas e sub-janelamento.
    2. Embeddings Neurais SpeechBrain (ECAPA-TDNN) ou Biometria Acústica local.
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
        # 1. Se o usuário configurar explicitamente Pyannote e houver token válido, tenta pyannote
        if settings.ENABLE_PYANNOTE and self.hf_token:
            try:
                logger.info("Tentando diarização com Pyannote...")
                return self._diarize_pyannote(audio_path, min_speakers, max_speakers)
            except Exception as e:
                logger.warning(f"Pyannote indisponível ({e}). Tentando motor SpeechBrain...")

        # 2. Diarização Neural SpeechBrain (ECAPA-TDNN) 100% Local (alta precisão)
        try:
            return self._diarize_speechbrain(audio_path, min_speakers, max_speakers)
        except Exception as e:
            logger.warning(f"SpeechBrain indisponível ({e}). Executando motor acústico Librosa...")
    def _diarize_speechbrain(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Diarizador neural de alta precisão baseado em SpeechBrain (ECAPA-TDNN).
        Extrai assinaturas vocais densas de 192 dimensões para cada trecho de fala
        e agrupa interlocutores por similaridade angular (cosseno).
        """
        import torch
        import librosa
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize

        classifier = get_speechbrain_classifier()

        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        duration = len(y) / sr

        if duration < 1.0:
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Locutor 1"}]

        # 1. Detecção e fatiamento de fala (VAD Adaptativo)
        segments = self._detect_speech_segments(y, sr)
        if not segments:
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Locutor 1"}]

        # 2. Extração de embeddings neurais em lotes
        valid_segments = []
        chunks = []
        min_samples = int(0.35 * sr)  # mínimo de 350ms para embedding de voz

        for start_sec, end_sec in segments:
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            chunk = y[start_idx:end_idx]
            if len(chunk) >= min_samples:
                valid_segments.append((start_sec, end_sec))
                chunks.append(chunk)

        if len(chunks) <= 1:
            return [
                {"start": round(s[0], 2), "end": round(s[1], 2), "speaker": "Locutor 1"}
                for s in (valid_segments or [(0.0, duration)])
            ]

        batch_size = 32
        embeddings_list = []
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                max_len = max(len(c) for c in batch_chunks)
                padded = np.zeros((len(batch_chunks), max_len), dtype=np.float32)
                for j, c in enumerate(batch_chunks):
                    padded[j, :len(c)] = c
                batch_tensor = torch.from_numpy(padded).to(classifier.device)
                emb_tensor = classifier.encode_batch(batch_tensor)  # (B, 1, 192)
                emb_np = emb_tensor.squeeze(1).cpu().numpy()
                embeddings_list.append(emb_np)

        raw_embeddings = np.vstack(embeddings_list)
        norm_embeddings = normalize(raw_embeddings)  # Espaço esférico (distância cosseno)
        n_samples = len(norm_embeddings)

        # 3. Determinação do número de locutores
        k = self._determine_k_neural(
            embeddings=norm_embeddings,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        logger.info(f"SpeechBrain: número ótimo de locutores determinado: k={k} ({n_samples} amostras).")

        # 4. Agrupamento Hierárquico Aglomerativo
        if k <= 1:
            labels = [0] * n_samples
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward"
            )
            labels = clusterer.fit_predict(norm_embeddings)

            # Reatribuir micro-clusters espúrios (< 2% das fatias) apenas se o usuário NÃO tiver fixado min_speakers
            target_k = min_speakers if (min_speakers and max_speakers and min_speakers == max_speakers) else None
            if target_k is None:
                from sklearn.metrics.pairwise import cosine_distances
                counts = np.bincount(labels)
                min_viable = max(3, int(0.02 * n_samples))
                for cluster_id, count in enumerate(counts):
                    if count < min_viable and len(set(labels)) > 1:
                        other_clusters = [c for c in range(len(counts)) if c != cluster_id and counts[c] >= min_viable]
                        if other_clusters:
                            spurious_indices = np.where(labels == cluster_id)[0]
                            for idx in spurious_indices:
                                emb = norm_embeddings[idx:idx+1]
                                best_other = min(
                                    other_clusters,
                                    key=lambda c: float(cosine_distances(emb, np.mean(norm_embeddings[labels == c], axis=0, keepdims=True))[0][0])
                                )
                                labels[idx] = best_other

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

        smoothed = self._smooth_segments(raw_diarized)
        logger.info(f"Diarização SpeechBrain concluída: {len(speaker_map)} locutores identificados em {len(smoothed)} segmentos.")
        return smoothed

    def _determine_k_neural(
        self,
        embeddings: np.ndarray,
        min_speakers: Optional[int],
        max_speakers: Optional[int]
    ) -> int:
        """Determina o número ótimo de oradores no espaço de embeddings neurais."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_distances

        n_samples = len(embeddings)
        if n_samples <= 2:
            return 1

        if min_speakers and max_speakers and min_speakers == max_speakers:
            return max(1, min(min_speakers, n_samples))
        if min_speakers and not max_speakers and min_speakers > 1:
            return max(1, min(min_speakers, n_samples))
        if max_speakers and max_speakers == 1:
            return 1

        # Testar se pode ser 1 locutor único (caso de microfone isolado)
        if not min_speakers or min_speakers == 1:
            try:
                c2 = AgglomerativeClustering(n_clusters=2, linkage="ward")
                l2 = c2.fit_predict(embeddings)
                c_0 = np.mean(embeddings[l2 == 0], axis=0, keepdims=True)
                c_1 = np.mean(embeddings[l2 == 1], axis=0, keepdims=True)
                c_0 = c_0 / max(float(np.linalg.norm(c_0)), 1e-6)
                c_1 = c_1 / max(float(np.linalg.norm(c_1)), 1e-6)
                dist_between_centroids = float(cosine_distances(c_0, c_1)[0][0])
                sil_2 = silhouette_score(embeddings, l2)

                # Se os clusters estão muito próximos ou a silhueta for fraca, trata-se de um único orador
                if dist_between_centroids < 0.28 or sil_2 < 0.12:
                    logger.info(f"Detector de locutor único (SpeechBrain): dist={dist_between_centroids:.3f}, sil={sil_2:.3f} -> 1 locutor.")
                    return 1
            except Exception as e:
                logger.debug(f"Erro ao testar k=1: {e}")

        min_k = max(2, min_speakers or 2)
        max_k = min(8, max_speakers or 8, n_samples - 1)
        if max_k < min_k:
            return min_k

        best_k = min_k
        best_score = -999.0
        min_cluster_size = max(2, int(0.015 * n_samples))

        for candidate_k in range(min_k, max_k + 1):
            try:
                clusterer = AgglomerativeClustering(n_clusters=candidate_k, linkage="ward")
                pred_labels = clusterer.fit_predict(embeddings)
                counts = np.bincount(pred_labels)
                if np.min(counts) < min_cluster_size:
                    continue
                sil = silhouette_score(embeddings, pred_labels)
                if sil > best_score:
                    best_score = sil
                    best_k = candidate_k
            except Exception:
                pass

        if best_score == -999.0:
            for candidate_k in range(min_k, max_k + 1):
                try:
                    clusterer = AgglomerativeClustering(n_clusters=candidate_k, linkage="ward")
                    pred_labels = clusterer.fit_predict(embeddings)
                    sil = silhouette_score(embeddings, pred_labels)
                    if sil > best_score:
                        best_score = sil
                        best_k = candidate_k
                except Exception:
                    pass

        return best_k

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
        - Extrai assinaturas vocais (MFCCs sem energia 0 + Formantes + Pitch F0 + Dinâmica).
        - Padroniza distribuição acústica e executa clustering aglomerativo com Ward linkage.
        """
        try:
            import librosa
            from sklearn.preprocessing import StandardScaler, normalize
            from sklearn.cluster import AgglomerativeClustering

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

            raw_arr = np.array(features)
            n_samples = len(raw_arr)

            # 3. Padronização Acústica e Normalização L2
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(raw_arr)
            features_norm = normalize(features_scaled)

            # 4. Determinação do número de interlocutores
            k = self._determine_k(
                features=features_norm,
                n_samples=n_samples,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

            # 5. Clustering hierárquico aglomerativo com Ward linkage
            if k <= 1:
                labels = [0] * n_samples
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=k,
                    linkage="ward"
                )
                labels = clusterer.fit_predict(features_norm)

            # 6. Mapeamento cronológico dos locutores
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

            # 7. Suavização temporal e fusão de fatias contíguas
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

    def _detect_speech_segments(self, y: np.ndarray, sr: int = 16000) -> List[Tuple[float, float]]:
        """
        Detecta intervalos de voz humana usando Silero VAD (Rede Neural de Alta Precisão).
        Se indisponível, recorre ao algoritmo acústico Librosa por energia (top_db).
        Aplica sub-janelamento dinâmico em turnos contínuos de fala para capturar trocas rápidas de interlocutores.
        """
        # Linha de base acústica via Librosa
        import librosa
        lib_intervals = librosa.effects.split(y, top_db=28, frame_length=2048, hop_length=512)
        if len(lib_intervals) == 0:
            lib_intervals = librosa.effects.split(y, top_db=20, frame_length=2048, hop_length=512)

        intervals = []
        # 1. Tentativa com Silero VAD (Neural ONNX do Faster-Whisper)
        try:
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            vad_opts = VadOptions(
                threshold=0.45,
                min_speech_duration_ms=250,
                max_speech_duration_s=float("inf"),
                min_silence_duration_ms=400,
                speech_pad_ms=200
            )
            y_float = y.astype(np.float32)
            timestamps = get_speech_timestamps(y_float, vad_options=vad_opts, sampling_rate=sr)
            # Usar Silero se detectou fatias de voz consistentes com o áudio
            if timestamps and (len(timestamps) >= 2 or len(lib_intervals) <= 1 or len(timestamps) >= len(lib_intervals)):
                intervals = [(int(t["start"]), int(t["end"])) for t in timestamps]
                logger.debug(f"Silero VAD detectou {len(intervals)} intervalos de voz.")
        except Exception as e_vad:
            logger.debug(f"Silero VAD falhou ({e_vad}). Recorrendo ao fallback Librosa...")

        # 2. Fallback acústico Librosa caso Silero VAD não encontre trechos suficientes
        if not intervals:
            intervals = list(lib_intervals)
            if len(intervals) == 0:
                return [(0.0, len(y) / sr)]

        # Mesclar pausas curtas (< 0.45s) e descartar trechos imperceptíveis (< 0.30s)
        merge_gap = int(0.45 * sr)
        min_len = int(0.30 * sr)
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
            # 1. MFCC 1..20 (exclui coeficiente 0 para ser invariante a volume)
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=21)[1:]
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Deltas temporais dos MFCCs
            mfcc_delta = librosa.feature.delta(mfcc)
            delta_mean = np.mean(mfcc_delta, axis=1)

            # 2. Contraste Espectral (6 bandas) - ressonâncias e formantes vocais
            contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr, n_bands=6)
            contrast_mean = np.mean(contrast, axis=1)
            contrast_std = np.std(contrast, axis=1)

            # 3. Centróide, Rolloff, Largura de Banda e Flatness (forma do espectro vocal)
            centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
            flatness = librosa.feature.spectral_flatness(y=chunk)
            spec_stats = np.array([
                np.mean(centroid), np.std(centroid),
                np.mean(rolloff), np.std(rolloff),
                np.mean(bandwidth), np.std(bandwidth),
                np.mean(flatness), np.std(flatness)
            ])

            # 4. Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(chunk)
            zcr_stats = np.array([np.mean(zcr), np.std(zcr)])

            # 5. Pitch F0 (frequência fundamental da voz via YIN: 65Hz a 400Hz)
            try:
                f0 = librosa.yin(chunk, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=256)
                voiced = f0[(f0 >= 65) & (f0 <= 400)]
                if len(voiced) > 0:
                    f0_stats = np.array([
                        float(np.median(voiced)),
                        float(np.std(voiced)),
                        float(len(voiced) / len(f0))
                    ])
                else:
                    f0_stats = np.array([0.0, 0.0, 0.0])
            except Exception:
                f0_stats = np.array([0.0, 0.0, 0.0])

            # Concatena vetor multimodal completo
            feat = np.hstack([
                mfcc_mean, mfcc_std, delta_mean,
                contrast_mean, contrast_std,
                spec_stats, zcr_stats, f0_stats
            ])
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
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
        """Determina o número ótimo de locutores via Ward clustering, Davies-Bouldin e Silhouette."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        # Caso o usuário tenha definido valor fixo de participantes
        if min_speakers and max_speakers and min_speakers == max_speakers:
            return max(1, min(min_speakers, n_samples))
        if min_speakers and not max_speakers:
            return max(1, min(min_speakers, n_samples))
        if max_speakers and not min_speakers and max_speakers == 1:
            return 1

        max_k_possible = min(8, n_samples - 1)
        if max_speakers:
            max_k_possible = min(max_speakers, max_k_possible)
        min_k_possible = max(1, min_speakers or 1) if (min_speakers == 1 or max_speakers == 1) else max(2, min_speakers or 2)

        if max_k_possible < min_k_possible:
            return min(min_k_possible, n_samples)
        if max_k_possible == min_k_possible:
            return min_k_possible

        best_k = min_k_possible
        best_score = -999.0

        # Primeira rodada: busca penalizando clusters menores que 2.5% das amostras (outliers)
        min_cluster_size = max(2, int(0.025 * n_samples))

        for candidate_k in range(min_k_possible, max_k_possible + 1):
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=candidate_k,
                    linkage="ward"
                )
                pred_labels = clusterer.fit_predict(features)
                if len(set(pred_labels)) > 1:
                    counts = np.bincount(pred_labels)
                    if np.min(counts) < min_cluster_size:
                        continue
                    sil = silhouette_score(features, pred_labels)
                    db = davies_bouldin_score(features, pred_labels)
                    score = sil / max(db, 0.001)
                    if score > best_score:
                        best_score = score
                        best_k = candidate_k
            except Exception:
                pass

        # Se todos foram filtrados pelo tamanho mínimo, avalia sem o filtro de outlier
        if best_score == -999.0:
            for candidate_k in range(min_k_possible, max_k_possible + 1):
                try:
                    clusterer = AgglomerativeClustering(
                        n_clusters=candidate_k,
                        linkage="ward"
                    )
                    pred_labels = clusterer.fit_predict(features)
                    if len(set(pred_labels)) > 1:
                        sil = silhouette_score(features, pred_labels)
                        db = davies_bouldin_score(features, pred_labels)
                        score = sil / max(db, 0.001)
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
