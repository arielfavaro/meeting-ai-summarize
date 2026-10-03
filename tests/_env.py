"""Ambiente isolado para os testes: dados em diretório temporário e raiz do projeto no sys.path."""
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TEST_DATA_DIR = Path(os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="meetingai-tests-")))
os.environ.setdefault("HF_HOME", str(TEST_DATA_DIR / "models_cache"))
