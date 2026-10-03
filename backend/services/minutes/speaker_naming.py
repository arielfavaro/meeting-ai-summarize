"""
Validação determinística das sugestões de nome feitas pelo LLM.

O LLM só PROPÕE; quem decide aplicar é este módulo, com regras verificáveis:
- o nome precisa aparecer na transcrição;
- auto-apresentação: a fala de evidência é do próprio locutor e contém o nome;
- chamado pelo nome: a fala de evidência é de OUTRO locutor, contém o nome, e a fala
  seguinte é do locutor sugerido (quem é chamado é quem responde);
- confiança mínima, unicidade de nomes e respeito a nomes definidos pelo usuário.
"""
import re
import unicodedata
from typing import Dict, List, Sequence

from backend.models.schemas import SpeakerSegment, SpeakerSuggestion
from backend.services.minutes.llm_schemas import LLMSpeakerName
from backend.services.speakers import is_label

MAX_NAME_WORDS = 4


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_name(text: str, name: str) -> bool:
    first = _norm(name).split(" ")[0] if name else ""
    if len(first) < 3:
        return False
    return re.search(rf"\b{re.escape(first)}\b", _norm(text)) is not None


def validate_speaker_suggestions(
    raw: Sequence[LLMSpeakerName],
    segments: Sequence[SpeakerSegment],
    speaker_map: Dict[str, str],
    sources: Dict[str, str],
    min_confidence: float = 0.75,
) -> List[SpeakerSuggestion]:
    by_id = {s.id: s for s in segments}
    order = {s.id: i for i, s in enumerate(segments)}
    known_ids = {s.speaker_id for s in segments}
    full_text = " ".join(s.text for s in segments)

    results: List[SpeakerSuggestion] = []
    for r in raw:
        name = " ".join((r.name or "").split()).strip(" .,:;\"'").title() if r.name else ""
        sug = SpeakerSuggestion(
            speaker_id=r.speaker_id, name=name, confidence=max(0.0, min(1.0, float(r.confidence))),
            kind=r.kind, evidence=[e for e in r.evidence if e in by_id],
        )

        def reject(reason: str) -> None:
            sug.applied, sug.reason = False, reason

        if r.speaker_id not in known_ids:
            reject("rótulo inexistente")
        elif not name or is_label(name) or len(name.split()) > MAX_NAME_WORDS:
            reject("nome inválido")
        elif not _contains_name(full_text, name):
            reject("nome não aparece na transcrição")
        elif sources.get(r.speaker_id) == "user":
            reject("nome já definido pelo usuário")
        elif sug.confidence < min_confidence:
            reject(f"confiança abaixo de {min_confidence:.2f}")
        elif sug.kind == "auto_apresentacao":
            ok = any(by_id[e].speaker_id == r.speaker_id and _contains_name(by_id[e].text, name) for e in sug.evidence)
            sug.applied, sug.reason = (True, "auto-apresentação") if ok else (False, "evidência não confirma auto-apresentação")
        elif sug.kind == "chamado_pelo_nome":
            ok = False
            for e in sug.evidence:
                seg = by_id[e]
                if seg.speaker_id == r.speaker_id or not _contains_name(seg.text, name):
                    continue
                nxt_idx = order[e] + 1
                if nxt_idx < len(segments) and segments[nxt_idx].speaker_id == r.speaker_id:
                    ok = True
                    break
            sug.applied, sug.reason = (True, "chamado pelo nome e respondeu") if ok else (False, "quem foi chamado não respondeu em seguida")
        else:
            reject("indício fraco")
        results.append(sug)

    # Unicidade: um nome só pode ir para um locutor, e não pode colidir com nome do usuário.
    user_names = {_norm(n) for sid, n in speaker_map.items() if sources.get(sid) == "user"}
    applied = [s for s in results if s.applied]
    for s in applied:
        key = _norm(s.name)
        if key in user_names:
            s.applied, s.reason = False, "nome já usado por outro locutor"
            continue
        rivals = [o for o in applied if o is not s and o.applied and _norm(o.name) == key]
        if rivals and any(o.confidence >= s.confidence for o in rivals):
            s.applied, s.reason = False, "nome sugerido para mais de um locutor"

    # Um locutor recebe no máximo um nome (o de maior confiança).
    best: Dict[str, SpeakerSuggestion] = {}
    for s in results:
        if s.applied:
            cur = best.get(s.speaker_id)
            if cur is None or s.confidence > cur.confidence:
                if cur:
                    cur.applied, cur.reason = False, "outra sugestão com maior confiança"
                best[s.speaker_id] = s
            else:
                s.applied, s.reason = False, "outra sugestão com maior confiança"
    return results
