"""
Tests for offline_ai_assistant.data.model_manager.

Uses a temporary directory for models_dir. download_model is mocked to avoid
network calls.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from offline_ai_assistant.data.model_manager import (
    ModelInfo,
    ModelManager,
    create_model_manager,
)


class TestModelInfo(unittest.TestCase):
    def test_model_info_has_expected_attributes(self):
        info = ModelInfo(
            id="test",
            name="Test Model",
            description="A test",
            size_gb=1.0,
            url="https://example.com/model.gguf",
            filename="model.gguf",
        )
        self.assertEqual(info.id, "test")
        self.assertEqual(info.filename, "model.gguf")
        self.assertEqual(info.quantization, "Q4_K_M")
        self.assertEqual(info.context_length, 4096)


class TestModelManagerInit(unittest.TestCase):
    def test_init_creates_models_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models"
            mgr = ModelManager(models_dir=path)
            self.assertTrue(path.exists())
            self.assertEqual(mgr.models_dir, path)

    def test_init_loads_metadata_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            meta_file = path / "models_metadata.json"
            meta_file.write_text('{"a.gguf": {"downloaded_at": "2024-01-01"}}')
            mgr = ModelManager(models_dir=path)
            self.assertEqual(mgr.metadata.get("a.gguf", {}).get("downloaded_at"), "2024-01-01")


class TestModelManagerGetAvailableModels(unittest.TestCase):
    def test_get_available_models_returns_list_of_model_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            models = mgr.get_available_models()
            self.assertIsInstance(models, list)
            self.assertGreater(len(models), 0)
            for m in models:
                self.assertIsInstance(m, ModelInfo)
                self.assertIsInstance(m.id, str)
                self.assertIsInstance(m.filename, str)
                self.assertIsInstance(m.url, str)


class TestModelManagerGetInstalledModels(unittest.TestCase):
    def test_get_installed_models_empty_when_no_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            installed = mgr.get_installed_models()
            self.assertEqual(installed, [])

    def test_get_installed_models_includes_known_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            # Create a file matching a known filename
            info = list(ModelManager.AVAILABLE_MODELS.values())[0]
            (path / info.filename).write_bytes(b"x" * 100)
            mgr = ModelManager(models_dir=path)
            installed = mgr.get_installed_models()
            self.assertEqual(len(installed), 1)
            self.assertEqual(installed[0]["filename"], info.filename)
            self.assertIn("path", installed[0])
            self.assertIn("size_mb", installed[0])
            self.assertIn("name", installed[0])


class TestModelManagerIsModelInstalled(unittest.TestCase):
    def test_unknown_model_id_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            self.assertFalse(mgr.is_model_installed("unknown-id"))

    def test_known_model_not_present_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            self.assertFalse(mgr.is_model_installed(model_id))

    def test_known_model_present_returns_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            info = ModelManager.AVAILABLE_MODELS[model_id]
            (path / info.filename).write_bytes(b"data")
            mgr = ModelManager(models_dir=path)
            self.assertTrue(mgr.is_model_installed(model_id))


class TestModelManagerDownloadModel(unittest.TestCase):
    def test_unknown_model_id_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            self.assertFalse(mgr.download_model("unknown-id"))

    def test_already_exists_returns_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            info = ModelManager.AVAILABLE_MODELS[model_id]
            (path / info.filename).write_bytes(b"existing")
            mgr = ModelManager(models_dir=path)
            self.assertTrue(mgr.download_model(model_id))

    def test_download_success_with_mocked_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            info = ModelManager.AVAILABLE_MODELS[model_id]
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"content-length": "100"}
            mock_response.iter_content = lambda **kw: [b"x" * 100]
            with patch("offline_ai_assistant.data.model_manager.requests.get", return_value=mock_response):
                mgr = ModelManager(models_dir=path)
                result = mgr.download_model(model_id)
            self.assertTrue(result)
            self.assertTrue((path / info.filename).exists())


class TestModelManagerDeleteModel(unittest.TestCase):
    def test_delete_model_removes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fname = "test-model.gguf"
            (path / fname).write_bytes(b"data")
            mgr = ModelManager(models_dir=path)
            result = mgr.delete_model(fname)
            self.assertTrue(result)
            self.assertFalse((path / fname).exists())

    def test_delete_model_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            self.assertFalse(mgr.delete_model("nonexistent.gguf"))


class TestModelManagerGetModelPath(unittest.TestCase):
    def test_unknown_model_id_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            self.assertIsNone(mgr.get_model_path("unknown-id"))

    def test_known_model_not_installed_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            self.assertIsNone(mgr.get_model_path(model_id))

    def test_known_model_installed_returns_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            info = ModelManager.AVAILABLE_MODELS[model_id]
            (path / info.filename).write_bytes(b"x")
            mgr = ModelManager(models_dir=path)
            result = mgr.get_model_path(model_id)
            self.assertIsNotNone(result)
            self.assertEqual(result, path / info.filename)


class TestModelManagerCalculateFileHash(unittest.TestCase):
    def test_calculate_file_hash_deterministic(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"content")
            path = Path(f.name)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                mgr = ModelManager(models_dir=Path(tmp))
                h1 = mgr.calculate_file_hash(path)
                h2 = mgr.calculate_file_hash(path)
                self.assertEqual(h1, h2)
                self.assertEqual(len(h1), 64)
        finally:
            path.unlink(missing_ok=True)


class TestModelManagerVerifyModel(unittest.TestCase):
    def test_unknown_model_id_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            self.assertFalse(mgr.verify_model("unknown-id"))

    def test_model_not_installed_with_sha256_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager(models_dir=Path(tmp))
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            # When sha256 is set but file missing, verify returns False
            original = ModelManager.AVAILABLE_MODELS[model_id]
            try:
                ModelManager.AVAILABLE_MODELS[model_id] = ModelInfo(
                    id=original.id,
                    name=original.name,
                    description=original.description,
                    size_gb=original.size_gb,
                    url=original.url,
                    filename=original.filename,
                    sha256="a" * 64,
                    quantization=original.quantization,
                    context_length=original.context_length,
                )
                self.assertFalse(mgr.verify_model(model_id))
            finally:
                ModelManager.AVAILABLE_MODELS[model_id] = original

    def test_model_no_sha256_returns_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model_id = list(ModelManager.AVAILABLE_MODELS.keys())[0]
            info = ModelManager.AVAILABLE_MODELS[model_id]
            (path / info.filename).write_bytes(b"x")
            mgr = ModelManager(models_dir=path)
            self.assertTrue(mgr.verify_model(model_id))


class TestCreateModelManager(unittest.TestCase):
    def test_create_model_manager_returns_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("offline_ai_assistant.data.model_manager.Config") as cfg:
                cfg.MODELS_DIR = Path(tmp)
                mgr = create_model_manager()
                self.assertIsInstance(mgr, ModelManager)
                self.assertEqual(mgr.models_dir, Path(tmp))


if __name__ == "__main__":
    unittest.main()
