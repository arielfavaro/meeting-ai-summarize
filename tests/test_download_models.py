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
             mock.patch.object(dm, "download_aligner", side_effect=lambda token=None: calls.append("al")), \
             mock.patch.object(dm, "download_pyannote", side_effect=lambda token: calls.append("py") or True):
            code = dm.main(["--whisper", "large-v3", "--hf-token", "hf_x"])
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["sb", "al", "py"])

    def test_aligner_already_present_is_skipped(self):
        target = self.root / "aligner"
        target.mkdir()
        for name in (*dm.ALIGNER_REQUIRED, "pytorch_model.bin"):
            (target / name).write_text("{}")
        snapshot = mock.Mock()
        with mock.patch.object(type(dm.settings), "alignment_model_dir", property(lambda s: target)), \
             mock.patch.dict(sys.modules, fake_hub(snapshot)):
            self.assertTrue(dm.download_aligner())
        snapshot.assert_not_called()

    def test_aligner_downloads_one_weight_format_without_language_model(self):
        target = self.root / "aligner"
        seen = {}

        def snapshot(repo, local_dir, token=None, allow_patterns=None, ignore_patterns=None):
            seen.update(allow=allow_patterns, ignore=ignore_patterns)
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            for name in ("config.json", "model.safetensors"):
                (Path(local_dir) / name).write_text("{}")

        modules = fake_hub(snapshot)
        modules["huggingface_hub"].list_repo_files = lambda repo, token=None: [
            "config.json", "vocab.json", "model.safetensors", "pytorch_model.bin", "language_model/lm.binary"]
        with mock.patch.object(type(dm.settings), "alignment_model_dir", property(lambda s: target)), \
             mock.patch.dict(sys.modules, modules):
            self.assertTrue(dm.download_aligner())
        self.assertEqual(seen["allow"], ["*.json", "model.safetensors"])
        self.assertEqual(seen["ignore"], ["language_model/*"])

    def test_only_aligner(self):
        calls = []
        with mock.patch.object(dm, "download_whisper", side_effect=lambda sizes: calls.append("wh")), \
             mock.patch.object(dm, "download_aligner", side_effect=lambda token=None: calls.append("al")):
            code = dm.main(["--skip-whisper", "--skip-speechbrain", "--skip-pyannote"])
        self.assertEqual((code, calls), (0, ["al"]))

    def test_whisper_already_in_cache_is_not_downloaded_again(self):
        calls = []

        def download_model(size, cache_dir=None, local_files_only=False):
            calls.append((size, local_files_only))
            if local_files_only and size == "large-v3":
                return "/cache/large-v3"
            if local_files_only:
                raise FileNotFoundError(size)
            return f"/cache/{size}"

        utils = types.ModuleType("faster_whisper.utils")
        utils.download_model = download_model
        with mock.patch.dict(sys.modules, {"faster_whisper": types.ModuleType("faster_whisper"),
                                           "faster_whisper.utils": utils}):
            self.assertTrue(dm.download_whisper(["large-v3", "medium"]))
        # large-v3 já estava no disco: só a verificação local; medium: verificação + download
        self.assertEqual(calls, [("large-v3", True), ("medium", True), ("medium", False)])

    def test_from_scratch_without_flags_checks_everything(self):
        calls = []
        with mock.patch.object(dm, "download_whisper", side_effect=lambda sizes: calls.append(("wh", sizes))), \
             mock.patch.object(dm, "download_speechbrain", side_effect=lambda token=None: calls.append("sb")), \
             mock.patch.object(dm, "download_aligner", side_effect=lambda token=None: calls.append("al")), \
             mock.patch.object(dm, "download_pyannote", side_effect=lambda token: calls.append("py")):
            code = dm.main(["--hf-token", "hf_x"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("wh", [dm.settings.WHISPER_MODEL_SIZE]), "sb", "al", "py"])

    def test_pyannote_on_disk_is_confirmed_even_without_token(self):
        self.py_dir.mkdir()
        (self.py_dir / "config.yaml").write_text("pipeline: x")
        calls = []
        with mock.patch.object(dm, "download_pyannote", side_effect=lambda token: calls.append(token)):
            dm.main(["--skip-whisper", "--skip-speechbrain", "--skip-aligner", "--hf-token", ""])
        self.assertEqual(calls, [""])

    def test_pyannote_already_present_is_skipped(self):
        self.py_dir.mkdir()
        (self.py_dir / "config.yaml").write_text("pipeline: x")
        snapshot = mock.Mock()
        with mock.patch.dict(sys.modules, fake_hub(snapshot)):
            self.assertTrue(dm.download_pyannote("hf_x"))
        snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
