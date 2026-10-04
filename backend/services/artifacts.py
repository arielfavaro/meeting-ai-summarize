"""
Checkpoints das etapas pesadas (faixas, diarização, transcrição) em JSON no disco local.

Cada artefato é identificado pela origem (file_id do upload) + tipo + parâmetros. Assim:
- "tentar novamente" após uma falha retoma do ponto onde parou;
- re-diarizar com outro número de pessoas reaproveita a transcrição (a etapa mais lenta);
- regerar a ata não depende de reprocessar áudio.
"""
import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SAFE_SOURCE = re.compile(r"^[0-9a-f-]{8,64}$")


class ArtifactStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def _source_dir(self, source_id: str) -> Path:
        stem = Path(source_id).stem
        if not SAFE_SOURCE.fullmatch(stem):
            raise ValueError(f"Origem inválida: {source_id}")
        return self.base_dir / stem

    @staticmethod
    def _key(kind: str, params: Dict[str, Any]) -> str:
        digest = hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        return f"{kind}-{digest}.json"

    def load(self, source_id: str, kind: str, params: Dict[str, Any]) -> Optional[Any]:
        path = self._source_dir(source_id) / self._key(kind, params)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("data")
        except (OSError, ValueError):
            logger.warning("Checkpoint corrompido ignorado: %s", path)
            return None

    def save(self, source_id: str, kind: str, params: Dict[str, Any], data: Any) -> None:
        directory = self._source_dir(source_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self._key(kind, params)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"kind": kind, "params": params, "data": data}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)  # escrita atômica: um checkpoint nunca fica pela metade

    def delete_source(self, source_id: str) -> None:
        directory = self._source_dir(source_id)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
