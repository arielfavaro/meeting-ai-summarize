"""Armazenamento de arquivos de áudio com validação de nomes (anti path traversal) e upload em streaming."""
import re
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".wma",
    ".mp4", ".webm", ".mkv", ".mov", ".avi",
}
FILE_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.[a-z0-9]{2,5}$")
UUID_STEM_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SAFE_NAME_PATTERN = re.compile(r"^[\w.-]{1,200}$")
CHUNK_SIZE = 1024 * 1024


class InvalidFileError(ValueError):
    pass


class FileTooLargeError(ValueError):
    pass


def _inside(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


class FileStore:
    def __init__(self, upload_dir: Path, processed_dir: Path, max_bytes: int):
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        self.max_bytes = max_bytes

    async def save_upload(self, upload: UploadFile) -> Tuple[str, int]:
        ext = Path(upload.filename or "").suffix.lower() or ".wav"
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileError(f"Extensão '{ext}' não suportada.")

        file_id = f"{uuid.uuid4()}{ext}"
        destination = self.upload_dir / file_id
        size = 0
        try:
            with open(destination, "wb") as buffer:
                while chunk := await upload.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise FileTooLargeError(
                            f"Arquivo excede o limite de {self.max_bytes // (1024 * 1024)}MB.")
                    buffer.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return file_id, size

    def upload_path(self, file_id: str) -> Path:
        if not FILE_ID_PATTERN.fullmatch(file_id or ""):
            raise InvalidFileError("Identificador de arquivo inválido.")
        path = self.upload_dir / file_id
        if not _inside(self.upload_dir, path) or not path.exists():
            raise FileNotFoundError(f"Arquivo {file_id} não encontrado.")
        return path

    def audio_path(self, filename: str) -> Path:
        if not SAFE_NAME_PATTERN.fullmatch(filename or "") or filename.startswith("."):
            raise InvalidFileError("Nome de arquivo inválido.")
        for base in (self.processed_dir, self.upload_dir):
            path = base / filename
            if _inside(base, path) and path.is_file():
                return path
        raise FileNotFoundError(filename)

    def delete_source_files(self, stem: str) -> int:
        """
        Remove o upload original e os WAVs derivados (mestre e faixas) de uma reunião.
        Só aceita o identificador UUID gerado pelo próprio sistema.
        """
        stem = Path(stem).stem
        if not UUID_STEM_PATTERN.fullmatch(stem):
            raise InvalidFileError("Identificador de arquivo inválido.")
        removed = 0
        candidates = [
            *self.upload_dir.glob(f"{stem}.*"),
            *self.processed_dir.glob(f"{stem}_16k.wav"),
            *self.processed_dir.glob(f"{stem}_track_*_16k.wav"),
        ]
        for path in candidates:
            if path.is_file() and (_inside(self.upload_dir, path) or _inside(self.processed_dir, path)):
                path.unlink(missing_ok=True)
                removed += 1
        return removed
