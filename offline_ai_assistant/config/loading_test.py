"""
Tests for offline_ai_assistant.config.loading.

Uses unittest and temporary directories so no real config or home dir is modified.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_ai_assistant.config.loading import Config, setup_logging, _apply_embedding_env_overrides


class TestConfigEnsureDirectories(unittest.TestCase):
    def test_ensure_directories_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Config.USER_DATA_DIR = tmp
            Config.DB_DIR = tmp / "db"
            Config.DOCS_DIR = tmp / "docs"
            Config.MODELS_DIR = tmp / "models"
            Config.LOGS_DIR = tmp / "logs"
            Config.ensure_directories()
            self.assertTrue(Config.DB_DIR.exists())
            self.assertTrue(Config.DOCS_DIR.exists())
            self.assertTrue(Config.MODELS_DIR.exists())
            self.assertTrue(Config.LOGS_DIR.exists())


class TestConfigGetSettingsDict(unittest.TestCase):
    def test_get_settings_dict_returns_dict_with_expected_keys(self):
        d = Config.get_settings_dict()
        self.assertIsInstance(d, dict)
        expected = {
            "chunk_size",
            "chunk_overlap",
            "top_k_retrieval",
            "llm_max_tokens",
            "llm_temperature",
            "embedding_model",
            "embedding_batch_size",
            "embedding_show_progress",
            "llm_model_path",
        }
        for key in expected:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_get_settings_dict_embedding_model_string(self):
        d = Config.get_settings_dict()
        self.assertIsInstance(d["embedding_model"], str)
        self.assertGreater(len(d["embedding_model"]), 0)

    def test_get_settings_dict_numeric_bounds(self):
        d = Config.get_settings_dict()
        self.assertGreaterEqual(d["chunk_size"], 64)
        self.assertLessEqual(d["chunk_size"], 4096)
        self.assertGreaterEqual(d["top_k_retrieval"], 1)
        self.assertLessEqual(d["top_k_retrieval"], 50)


class TestConfigUpdateSettings(unittest.TestCase):
    def setUp(self):
        self.saved = {
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "top_k_retrieval": Config.TOP_K_RETRIEVAL,
            "embedding_model": Config.EMBEDDING_MODEL_NAME,
        }

    def tearDown(self):
        Config.CHUNK_SIZE = self.saved["chunk_size"]
        Config.CHUNK_OVERLAP = self.saved["chunk_overlap"]
        Config.TOP_K_RETRIEVAL = self.saved["top_k_retrieval"]
        Config.EMBEDDING_MODEL_NAME = self.saved["embedding_model"]

    def test_update_settings_valid_updates_attrs(self):
        Config.update_settings(
            {"chunk_size": 256, "chunk_overlap": 25, "embedding_model": "all-mpnet-base-v2"},
            save=False,
        )
        self.assertEqual(Config.CHUNK_SIZE, 256)
        self.assertEqual(Config.CHUNK_OVERLAP, 25)
        self.assertEqual(Config.EMBEDDING_MODEL_NAME, "all-mpnet-base-v2")

    def test_update_settings_clamps_out_of_range(self):
        Config.update_settings({"chunk_size": 10, "top_k_retrieval": 100}, save=False)
        self.assertEqual(Config.CHUNK_SIZE, 64)  # clamped to min
        self.assertEqual(Config.TOP_K_RETRIEVAL, 50)  # clamped to max

    def test_update_settings_ignores_invalid_values(self):
        before = Config.CHUNK_SIZE
        Config.update_settings({"chunk_size": "not_a_number"}, save=False)
        self.assertEqual(Config.CHUNK_SIZE, before)


class TestConfigLoadAndSave(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp = Path(self.tmp)
        self.config_file = self.tmp / "config.json"
        self.orig_user_data = Config.USER_DATA_DIR
        self.orig_config_file = Config.CONFIG_FILE
        self.orig_db = Config.DB_DIR
        self.orig_docs = Config.DOCS_DIR
        self.orig_models = Config.MODELS_DIR
        self.orig_logs = Config.LOGS_DIR
        Config.USER_DATA_DIR = self.tmp
        Config.CONFIG_FILE = self.config_file
        Config.DB_DIR = self.tmp / "db"
        Config.DOCS_DIR = self.tmp / "docs"
        Config.MODELS_DIR = self.tmp / "models"
        Config.LOGS_DIR = self.tmp / "logs"

    def tearDown(self):
        Config.USER_DATA_DIR = self.orig_user_data
        Config.CONFIG_FILE = self.orig_config_file
        Config.DB_DIR = self.orig_db
        Config.DOCS_DIR = self.orig_docs
        Config.MODELS_DIR = self.orig_models
        Config.LOGS_DIR = self.orig_logs
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_config_when_file_missing_does_not_raise(self):
        self.assertFalse(Config.CONFIG_FILE.exists())
        Config.load_config()
        # Should not raise; env overrides still applied
        self.assertIsNotNone(Config.EMBEDDING_BATCH_SIZE)

    def test_save_config_writes_file(self):
        Config.CHUNK_SIZE = 256
        Config.save_config()
        self.assertTrue(Config.CONFIG_FILE.exists())
        with open(Config.CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("chunk_size"), 256)

    def test_load_config_reads_saved_file(self):
        payload = {"chunk_size": 400, "chunk_overlap": 40, "embedding_model": "all-MiniLM-L6-v2"}
        Config.USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(Config.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        Config.load_config()
        self.assertEqual(Config.CHUNK_SIZE, 400)
        self.assertEqual(Config.CHUNK_OVERLAP, 40)

    def test_load_config_invalid_json_does_not_crash(self):
        Config.USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(Config.CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write("not valid json {")
        Config.load_config()
        # Config unchanged or reset to defaults
        self.assertIsInstance(Config.CHUNK_SIZE, int)


class TestConfigResetToDefaults(unittest.TestCase):
    def test_reset_to_defaults_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Config.USER_DATA_DIR = tmp
            Config.CONFIG_FILE = tmp / "config.json"
            Config.DB_DIR = tmp / "db"
            Config.DOCS_DIR = tmp / "docs"
            Config.MODELS_DIR = tmp / "models"
            Config.LOGS_DIR = tmp / "logs"
            Config.CHUNK_SIZE = 999
            Config.reset_to_defaults()
            self.assertEqual(Config.CHUNK_SIZE, 512)
            self.assertTrue(Config.CONFIG_FILE.exists())


class TestSetupLogging(unittest.TestCase):
    def test_setup_logging_returns_logger(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Config.USER_DATA_DIR = tmp
            Config.LOGS_DIR = tmp / "logs"
            Config.LOG_FILE = tmp / "logs" / "app.log"
            Config.ensure_directories()
            logger = setup_logging()
            self.assertIsNotNone(logger)
            self.assertEqual(logger.name, "OfflineAIAssistant")


class TestEmbeddingEnvOverrides(unittest.TestCase):
    def test_apply_embedding_env_overrides_batch_size(self):
        with patch.dict(os.environ, {"OFFLINE_AI_EMBEDDING_BATCH_SIZE": "16"}, clear=False):
            from importlib import reload
            from offline_ai_assistant.config import loading
            reload(loading)
            loading._apply_embedding_env_overrides()
            self.assertEqual(loading.Config.EMBEDDING_BATCH_SIZE, 16)

    def test_apply_embedding_env_overrides_show_progress_false(self):
        with patch.dict(os.environ, {"OFFLINE_AI_EMBEDDING_SHOW_PROGRESS": "0"}, clear=False):
            from importlib import reload
            from offline_ai_assistant.config import loading
            reload(loading)
            loading._apply_embedding_env_overrides()
            self.assertFalse(loading.Config.EMBEDDING_SHOW_PROGRESS)


if __name__ == "__main__":
    unittest.main()
