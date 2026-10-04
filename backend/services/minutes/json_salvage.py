"""
Aproveita uma resposta JSON cortada (limite de tokens ou geração abortada por laço).

Percorre o texto registrando pontos de corte seguros — logo após um valor completo — e
fecha os colchetes/chaves que ficaram abertos. O primeiro corte (do fim para o começo)
que vira JSON válido é usado. Campos de lista que ficaram faltando são preenchidos vazios.
"""
import json
import typing
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

_CLOSERS = {"{": "}", "[": "]"}


def salvage_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    stack: List[str] = []
    cuts: List[Tuple[int, Tuple[str, ...]]] = []
    in_string = escape = False

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if not stack:
                cuts.append((i + 1, ()))
                break
            cuts.append((i + 1, tuple(stack)))
        elif ch == "," and stack:
            cuts.append((i, tuple(stack)))  # corta antes da vírgula: o valor anterior está completo

    for cut, open_stack in reversed(cuts):
        # Só corta onde o que sobra continua íntegro: entre itens de uma lista ou entre
        # chaves do objeto raiz — nunca no meio de um item (que ficaria sem campos).
        if open_stack and open_stack[-1] != "[" and len(open_stack) != 1:
            continue
        candidate = s[:cut].rstrip().rstrip(",") + "".join(_CLOSERS[c] for c in reversed(open_stack))
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def fill_missing_fields(data: Dict[str, Any], model: Type[BaseModel]) -> Dict[str, Any]:
    """Completa campos de topo ausentes (listas → [], textos → "") para a validação passar."""
    out = dict(data)
    for name, field in model.model_fields.items():
        if name in out:
            continue
        origin = typing.get_origin(field.annotation)
        if origin in (list, List):
            out[name] = []
        elif field.annotation is str:
            out[name] = ""
    return out


def substantive_items(obj: BaseModel) -> int:
    """Quantos itens úteis a resposta trouxe (para não aceitar um JSON 'vazio' como resultado)."""
    keys = ("objectives", "topics", "decisions", "action_items", "open_points", "risks")
    return sum(len(getattr(obj, k, []) or []) for k in keys)
