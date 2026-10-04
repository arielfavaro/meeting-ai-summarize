"""Script de download único dos modelos (sem rede: downloads simulados)."""
import _env  # noqa: F401

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_previous = os.environ.get("HF_HUB_OFFLINE")
from backend.scripts import download_models as dm  # noqa: E402  (o módulo força HF_HUB_OFFLINE=0)
if _previous is None:
    os.environ.pop("HF_HUB_OFFLINE", None)
else:
    os.environ["HF_HUB_OFFLINE"] = _previous


def fake_hub(snapshot):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot
    errors = types.ModuleType("huggingface_hub.errors")
    errors.GatedRepoError = type("GatedRepoError", (Exception,), {})
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    return {"huggingface_hub": hub, "huggingface_hub.errors": errors}


class TestDownloadModels(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="meetingai-models-"))
        self.sb_dir = self.root / "speechbrain_ecapa"
        self.py_dir = self.root / "pyannote"
        patcher = mock.patch.multiple(type(dm.settings), speechbrain_model_dir=property(lambda s: self.sb_dir),
                                      pyannote_model_dir=property(lambda s: self.py_dir))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_existing_speechbrain_symlinks_are_reused_without_download(self):
        cache = self.root / "hub"
        cache.mkdir()
        self.sb_dir.mkdir()
        for name in dm.SPEECHBRAIN_REQUIRED:
            (cache / name).write_text("x")
            (self.sb_dir / name).symlink_to(cache / name)  # layout deixado pelo SpeechBrain antigo
        snapshot = mock.Mock()
        with mock.patch.dict(sys.modules, fake_hub(snapshot)):
            self.assertTrue(dm.download_speechbrain())
        snapshot.assert_not_called()

    def test_broken_symlinks_are_replaced_before_download(self):
        """Reproduz o SameFileError: links do diretório local apontando para o cache."""
        self.sb_dir.mkdir()
        (self.sb_dir / "hyperparams.yaml").symlink_to(self.root / "nao-existe.yaml")

        def snapshot(repo, local_dir, token=None):
            target = Path(local_dir)
            self.assertFalse(any(p.is_symlink() for p in target.iterdir()))
            for name in dm.SPEECHBRAIN_REQUIRED:
                (target / name).write_text("ok")
            return local_dir

        with mock.patch.dict(sys.modules, fake_hub(snapshot)):
            self.assertTrue(dm.download_speechbrain(token="hf_x"))
        self.assertTrue(dm._files_ready(self.sb_dir, dm.SPEECHBRAIN_REQUIRED))

    def test_one_failure_does_not_stop_the_others(self):
        calls = []
        with mock.patch.object(dm, "download_whisper", side_effect=RuntimeError("rede caiu")), \
             mock.patch.object(dm, "download_speechbrain", side_effect=lambda token=None: calls.append("sb")), \
             mock.patch.object(dm, "download_pyannote", side_effect=lambda token: calls.append("py") or True):
            code = dm.main(["--whisper", "large-v3", "--hf-token", "hf_x"])
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["sb", "py"])

    def test_pyannote_already_present_is_skipped(self):
        self.py_dir.mkdir()
        (self.py_dir / "config.yaml").write_text("pipeline: x")
        snapshot = mock.Mock()
        with mock.patch.dict(sys.modules, fake_hub(snapshot)):
            self.assertTrue(dm.download_pyannote("hf_x"))
        snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
