"""
Fachada das assinaturas de voz para o pipeline e para a API.

Fluxo no processamento:
1. depois da diarização, extrai uma voz por (faixa, locutor);
2. funde rótulos com vozes muito parecidas (mesma faixa: limiar mais alto; faixas diferentes:
   limiar próprio, porque microfone e áudio da chamada soam diferentes);
3. depois do alinhamento, compara as vozes finais com a biblioteca e nomeia quem é conhecido.

Tudo local: o modelo (SpeechBrain ECAPA) é lido do disco e as assinaturas ficam no SQLite.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from backend import clock
from backend.services.voice.clustering import ClusterResult, cluster_voices
from backend.services.voice.embedder import (
    EcapaEncoder, EncodeFn, SpeakerVoice, combine, embed_speakers, load_audio, spans_by_speaker,
)
from backend.services.voice.library import Voiceprint, VoiceMatch, enroll_vector, match_voices

logger = logging.getLogger(__name__)

TrackKey = Tuple[int, str]  # (índice da faixa, rótulo local do diarizador)


@dataclass
class VoiceConfig:
    enabled: bool = True
    merge_threshold: float = 0.70        # mesma faixa: fundir rótulos da mesma pessoa
    cross_track_threshold: float = 0.60  # faixas diferentes (microfone x chamada)
    library_enabled: bool = True
    match_threshold: float = 0.65        # biblioteca: reconhecer uma pessoa conhecida
    match_margin: float = 0.08           # folga mínima sobre a 2ª pessoa mais parecida

    @classmethod
    def from_settings(cls, s) -> "VoiceConfig":
        return cls(enabled=s.VOICE_FEATURES_ENABLED, merge_threshold=s.VOICE_MERGE_THRESHOLD,
                   cross_track_threshold=s.VOICE_CROSS_TRACK_THRESHOLD, library_enabled=s.VOICE_LIBRARY_ENABLED,
                   match_threshold=s.VOICE_MATCH_THRESHOLD, match_margin=s.VOICE_MATCH_MARGIN)


class VoiceService:
    def __init__(self, config: VoiceConfig, store=None, encoder: Optional[EncodeFn] = None):
        self.cfg = config
        self.store = store
        self.encoder = encoder or EcapaEncoder()

    # ----------------------------------------------------------- disponibilidade
    def unavailable_reason(self) -> Optional[str]:
        if not self.cfg.enabled:
            return "desativado (VOICE_FEATURES_ENABLED=false)"
        check = getattr(self.encoder, "unavailable_reason", None)
        return check() if check else None

    @property
    def library_active(self) -> bool:
        return self.cfg.library_enabled and self.store is not None and self.unavailable_reason() is None

    def release(self) -> None:
        release = getattr(self.encoder, "release", None)
        if release:
            release()

    # ------------------------------------------------------------------ vozes
    def voices_for_turns(self, audio_path: Path, turns: Iterable[Dict[str, Any]]) -> Dict[str, SpeakerVoice]:
        return embed_speakers(load_audio(audio_path), spans_by_speaker(turns), self.encoder)

    def voices_for_segments(self, audio_path: Path, segments: Sequence[Any]) -> Dict[str, SpeakerVoice]:
        turns = [{"start": s.start, "end": s.end, "speaker": s.speaker_id} for s in segments]
        return self.voices_for_turns(audio_path, turns)

    # ----------------------------------------------------------------- fusão
    def plan_merges(self, voices: Dict[TrackKey, SpeakerVoice], single_speaker_tracks: Set[int],
                    allow_within_track: bool = True) -> ClusterResult:
        """
        Restrições: dois canais individuais diferentes (dois microfones) são pessoas diferentes;
        com `allow_within_track=False` (número de pessoas fixado pelo usuário), rótulos da mesma
        faixa nunca se fundem.
        """
        def cannot_link(a: Hashable, b: Hashable) -> bool:
            (ta, _), (tb, _) = a, b
            if ta == tb:
                return not allow_within_track
            return ta in single_speaker_tracks and tb in single_speaker_tracks

        def threshold_for(a: Hashable, b: Hashable) -> float:
            return self.cfg.merge_threshold if a[0] == b[0] else self.cfg.cross_track_threshold

        return cluster_voices(voices, threshold=min(self.cfg.merge_threshold, self.cfg.cross_track_threshold),
                              cannot_link=cannot_link, threshold_for=threshold_for)

    # ------------------------------------------------------------- biblioteca
    def identify(self, voices: Dict[str, SpeakerVoice]) -> List[VoiceMatch]:
        if not self.library_active:
            return []
        prints = self.store.list()
        if not prints:
            return []
        return match_voices(voices, prints, self.cfg.match_threshold, self.cfg.match_margin)

    def enroll(self, name: str, voice: SpeakerVoice) -> Voiceprint:
        existing = self.store.get_by_name(name)
        vp = enroll_vector(existing, voice, name.strip(), clock.now().isoformat(timespec="seconds"))
        self.store.save(vp)
        logger.info("Voz de '%s' %s (%.0fs de fala, %d reunião(ões)).", vp.name,
                    "atualizada" if existing else "cadastrada", vp.seconds, vp.samples)
        return vp

    def enroll_from_meeting(self, audio_path: Path, segments: Sequence[Any], names: Dict[str, str]) -> List[str]:
        """Cadastra/atualiza as vozes dos locutores nomeados pelo usuário. Retorna os nomes cadastrados."""
        if not self.library_active or not names:
            return []
        voices = self.voices_for_segments(audio_path, [s for s in segments if s.speaker_id in names])
        done = []
        for sid, name in names.items():
            if sid in voices:
                self.enroll(name, voices[sid])
                done.append(name)
        return done


def relabel_with_groups(diarizations: Dict[int, List[Dict[str, Any]]], groups: Dict[TrackKey, int],
                        voices: Dict[TrackKey, SpeakerVoice]) -> Tuple[Dict[int, List[Dict[str, Any]]],
                                                                       Dict[str, SpeakerVoice]]:
    """
    Aplica a fusão: cada grupo vira um rótulo global "Locutor N" (ordem da 1ª fala considerando
    todas as faixas). Rótulos sem voz suficiente para comparar ficam como locutores próprios.
    Retorna as diarizações reescritas e a voz combinada de cada rótulo global.
    """
    group_of: Dict[TrackKey, Hashable] = {}
    first_start: Dict[Hashable, float] = {}
    for t, turns in diarizations.items():
        for turn in turns:
            key = (t, turn["speaker"])
            gid = groups.get(key, ("solo",) + key)
            group_of[key] = gid
            first_start[gid] = min(first_start.get(gid, float("inf")), float(turn["start"]))

    label_of = {gid: f"Locutor {i + 1}" for i, gid in enumerate(sorted(first_start, key=first_start.get))}
    relabeled = {t: [{**turn, "speaker": label_of[group_of[(t, turn["speaker"])]]} for turn in turns]
                 for t, turns in diarizations.items()}

    members: Dict[str, List[SpeakerVoice]] = {}
    for key, gid in group_of.items():
        if key in voices:
            members.setdefault(label_of[gid], []).append(voices[key])
    label_voices = {label: combine(vs) for label, vs in members.items()}
    return relabeled, {k: v for k, v in label_voices.items() if v is not None}
