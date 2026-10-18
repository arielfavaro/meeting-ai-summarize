"""
Fusão de locutores por assinatura de voz.

O diarizador às vezes divide uma pessoa em dois rótulos (mudança de tom, ruído, trechos
longos), e numa gravação multi-faixa cada faixa é diarizada separadamente: a mesma pessoa
pode aparecer no microfone E no áudio da chamada. Aqui os rótulos são agrupados por
similaridade de cosseno entre as vozes (agrupamento aglomerativo com limiar), respeitando
restrições "não podem ser a mesma pessoa".
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional

import numpy as np

from backend.services.voice.embedder import SpeakerVoice, unit

CannotLink = Callable[[Hashable, Hashable], bool]


@dataclass
class _Cluster:
    members: List[Hashable]
    weighted: np.ndarray            # soma(vetor * segundos)
    seconds: float

    @property
    def centroid(self) -> np.ndarray:
        return unit(self.weighted)


@dataclass
class MergeDecision:
    kept: Hashable
    merged: Hashable
    similarity: float


@dataclass
class ClusterResult:
    groups: Dict[Hashable, int]                       # chave -> índice do grupo
    decisions: List[MergeDecision] = field(default_factory=list)


def cluster_voices(voices: Dict[Hashable, SpeakerVoice], threshold: float,
                   cannot_link: Optional[CannotLink] = None,
                   threshold_for: Optional[Callable[[Hashable, Hashable], float]] = None) -> ClusterResult:
    """
    Agrupamento aglomerativo: a cada passo une o par de grupos com centróides mais parecidos,
    enquanto a similaridade for >= limiar e nenhum par de membros violar `cannot_link`.
    `threshold_for(a, b)` permite limiares diferentes (ex.: mesma faixa x faixas diferentes).
    """
    clusters = [_Cluster([k], v.vector * max(v.seconds, 1e-3), max(v.seconds, 1e-3)) for k, v in voices.items()]
    decisions: List[MergeDecision] = []

    def allowed(a: _Cluster, b: _Cluster) -> bool:
        return not cannot_link or not any(cannot_link(x, y) for x in a.members for y in b.members)

    def required(a: _Cluster, b: _Cluster) -> float:
        if not threshold_for:
            return threshold
        return max(threshold_for(x, y) for x in a.members for y in b.members)

    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                sim = float(np.dot(a.centroid, b.centroid))
                if sim < required(a, b) or not allowed(a, b):
                    continue
                if best is None or sim > best[0]:
                    best = (sim, i, j)
        if best is None:
            break
        sim, i, j = best
        a, b = clusters[i], clusters[j]
        kept, merged = (a, b) if a.seconds >= b.seconds else (b, a)
        decisions.append(MergeDecision(kept.members[0], merged.members[0], sim))
        clusters[i] = _Cluster(kept.members + merged.members, a.weighted + b.weighted, a.seconds + b.seconds)
        del clusters[j]

    groups = {key: idx for idx, c in enumerate(clusters) for key in c.members}
    return ClusterResult(groups, decisions)
