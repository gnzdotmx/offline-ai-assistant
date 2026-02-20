"""Tests for offline_ai_assistant.llm package re-exports (LocalLLM, LLMManager, create_llm)."""

import unittest


class TestLLMPackage(unittest.TestCase):
    """Test that offline_ai_assistant.llm exposes LocalLLM, LLMManager, create_llm."""

    def test_local_llm_importable(self):
        from offline_ai_assistant.llm import LocalLLM
        from offline_ai_assistant.llm.local_llm import LocalLLM as PkgLocalLLM
        self.assertIs(LocalLLM, PkgLocalLLM)

    def test_llm_manager_importable(self):
        from offline_ai_assistant.llm import LLMManager
        from offline_ai_assistant.llm.local_llm import LLMManager as PkgLLMManager
        self.assertIs(LLMManager, PkgLLMManager)

    def test_create_llm_importable(self):
        from offline_ai_assistant.llm import create_llm
        from offline_ai_assistant.llm.local_llm import create_llm as pkg_create_llm
        self.assertIs(create_llm, pkg_create_llm)

    def test_all_exports(self):
        import offline_ai_assistant.llm as llm_module
        self.assertIn("LocalLLM", llm_module.__all__)
        self.assertIn("LLMManager", llm_module.__all__)
        self.assertIn("create_llm", llm_module.__all__)
        self.assertEqual(len(llm_module.__all__), 3)


if __name__ == "__main__":
    unittest.main()
