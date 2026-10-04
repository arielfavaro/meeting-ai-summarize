"""Contrato dos motores de diarização (Strategy). Todos rodam localmente, sem rede."""
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

Turn = Dict[str, Any]          # {"start": float, "end": float, "speaker": "Locutor N"}
LogCallback = Callable[..., None]  # fn(mensagem, nivel="info")


class DiarizationEngine(Protocol):
    name: str
    label: str

    def unavailable_reason(self) -> Optional[str]:
        """None se o motor pode rodar agora; senão o motivo (modelo não baixado, biblioteca ausente...)."""

    def diarize(self, audio_path: Path, min_speakers: Optional[int], max_speakers: Optional[int],
                log: LogCallback) -> List[Turn]: ...

    def release(self) -> None:
        """Libera modelos da memória (VRAM)."""


def relabel_by_first_appearance(turns: List[Turn]) -> List[Turn]:
    """Rótulos estáveis 'Locutor N' na ordem em que cada voz aparece pela primeira vez."""
    mapping: Dict[Any, str] = {}
    out = []
    for t in sorted(turns, key=lambda x: (x["start"], x["end"])):
        if t["speaker"] not in mapping:
            mapping[t["speaker"]] = f"Locutor {len(mapping) + 1}"
        out.append({"start": round(float(t["start"]), 2), "end": round(float(t["end"]), 2),
                    "speaker": mapping[t["speaker"]]})
    return out


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
