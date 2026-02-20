"""
Tests for offline_ai_assistant.model_manager (compatibility shim).

Verifies that the package-root model_manager module re-exports the expected
symbols from data.model_manager.
"""

import unittest


class TestModelManagerShim(unittest.TestCase):
    """Test that the root model_manager module exposes the same API as data.model_manager."""

    def test_model_manager_importable(self):
        from offline_ai_assistant.model_manager import ModelManager
        from offline_ai_assistant.data.model_manager import ModelManager as DataModelManager
        self.assertIs(ModelManager, DataModelManager)

    def test_model_info_importable(self):
        from offline_ai_assistant.model_manager import ModelInfo
        from offline_ai_assistant.data.model_manager import ModelInfo as DataModelInfo
        self.assertIs(ModelInfo, DataModelInfo)

    def test_create_model_manager_importable(self):
        from offline_ai_assistant.model_manager import create_model_manager
        from offline_ai_assistant.data.model_manager import create_model_manager as data_create_model_manager
        self.assertIs(create_model_manager, data_create_model_manager)

    def test_all_exports(self):
        import offline_ai_assistant.model_manager as model_manager_module
        self.assertIn("ModelManager", model_manager_module.__all__)
        self.assertIn("ModelInfo", model_manager_module.__all__)
        self.assertIn("create_model_manager", model_manager_module.__all__)
        self.assertEqual(len(model_manager_module.__all__), 3)


if __name__ == "__main__":
    unittest.main()
