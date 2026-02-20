"""
Tests for offline_ai_assistant.llm.local_llm.

Uses mocks for Llama and model file so no real model or llama-cpp is required.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from offline_ai_assistant.core.models import GenerationConfig


def _make_mock_model():
    """Return a MagicMock that mimics Llama model for generate/tokenize/detokenize."""
    m = MagicMock()
    m.tokenize.return_value = [1, 2, 3]
    m.detokenize.return_value = b"hello world"
    m.n_vocab.return_value = 32000
    m.n_embd = 4096
    return m


def _make_llm_loaded(model_path=None):
    """Create LocalLLM with mocked _load_model so self.model is a mock (no real load)."""
    from offline_ai_assistant.llm.local_llm import LocalLLM

    with patch.object(LocalLLM, "_load_model") as load_mock:
        with patch("offline_ai_assistant.llm.local_llm.Llama", MagicMock()):
            with patch("offline_ai_assistant.llm.local_llm.Config") as cfg:
                cfg.LLM_MODEL_PATH = Path("/tmp/fake.gguf")
                cfg.LLM_CONTEXT_LENGTH = 2048
                cfg.LLM_N_GPU_LAYERS = 0
                cfg.LLM_N_BATCH = 512
                cfg.LLM_N_THREADS = 2
                cfg.PROMPT_TEMPLATES = {"default": "Context:\n{context}\n\nQuestion: {question}"}
                if model_path is None:
                    model_path = Path("/tmp/fake.gguf")
                with patch.object(model_path, "exists", return_value=True):
                    with patch.object(model_path, "stat") as stat:
                        stat.return_value = MagicMock(st_size=1024**3)
                        load_mock.side_effect = lambda: setattr(
                            next(
                                (x for x in [LocalLLM.__dict__.get("__init__")] if x),
                                "_llm_self",
                                None,
                            )
                        )
                        llm = LocalLLM(model_path=model_path)
                        llm.model = _make_mock_model()
                        llm.model_info = {"context_length": 2048}
    return llm


class TestLocalLLMDependencies(unittest.TestCase):
    def test_init_raises_when_llama_not_installed(self):
        from offline_ai_assistant.llm.local_llm import LocalLLM

        with patch("offline_ai_assistant.llm.local_llm.Llama", None):
            with patch("offline_ai_assistant.llm.local_llm.Config") as cfg:
                cfg.LLM_MODEL_PATH = Path("/tmp/x.gguf")
            with self.assertRaises(RuntimeError) as ctx:
                LocalLLM(model_path=Path("/tmp/x.gguf"))
            self.assertIn("llama-cpp-python", str(ctx.exception))


class TestLocalLLMModelLoad(unittest.TestCase):
    def test_init_raises_when_model_file_not_found(self):
        from offline_ai_assistant.llm.local_llm import LocalLLM

        with patch("offline_ai_assistant.llm.local_llm.Llama", MagicMock()):
            with patch("offline_ai_assistant.llm.local_llm.Config") as cfg:
                cfg.LLM_MODEL_PATH = Path("/nonexistent.gguf")
            with self.assertRaises(FileNotFoundError) as ctx:
                LocalLLM(model_path=Path("/nonexistent.gguf"))
            self.assertIn("not found", str(ctx.exception))


class TestLocalLLMWithMockModel(unittest.TestCase):
    """Tests that require a loaded mock model (no real Llama)."""

    def setUp(self):
        from offline_ai_assistant.llm.local_llm import LocalLLM

        with patch.object(LocalLLM, "_load_model"):
            with patch("offline_ai_assistant.llm.local_llm.Llama", MagicMock()):
                with patch("offline_ai_assistant.llm.local_llm.Config") as cfg:
                    cfg.LLM_MODEL_PATH = Path("/tmp/fake.gguf")
                    cfg.LLM_CONTEXT_LENGTH = 2048
                    cfg.LLM_N_GPU_LAYERS = 0
                    cfg.LLM_N_BATCH = 512
                    cfg.LLM_N_THREADS = 2
                    cfg.PROMPT_TEMPLATES = {"default": "{context}\n\n{question}"}
                    self.llm = LocalLLM(model_path=Path("/tmp/fake.gguf"))
                    self.llm.model = _make_mock_model()
                    self.llm.model_info = {"context_length": 2048}

    def test_generate_when_model_none_raises(self):
        self.llm.model = None
        with self.assertRaises(RuntimeError) as ctx:
            next(self.llm.generate("Hi"))
        self.assertIn("not loaded", str(ctx.exception))

    def test_generate_complete_returns_non_streaming_response(self):
        self.llm.model.return_value = {"choices": [{"text": "Hello."}]}
        out = self.llm.generate_complete("Hi", GenerationConfig(stream=False))
        self.assertEqual(out, "Hello.")

    def test_count_tokens_when_model_none_uses_word_split(self):
        self.llm.model = None
        self.assertEqual(self.llm.count_tokens("one two three"), 3)

    def test_count_tokens_uses_model_tokenize_when_loaded(self):
        self.llm.model.tokenize.return_value = [1, 2, 3, 4, 5]
        self.assertEqual(self.llm.count_tokens("hello"), 5)
        self.llm.model.tokenize.assert_called_once()

    def test_create_rag_prompt_formats_context_and_question(self):
        prompt = self.llm.create_rag_prompt(
            "What is it?",
            ["Chunk one.", "Chunk two."],
            template="Context:\n{context}\n\nQuestion: {question}",
        )
        self.assertIn("Chunk one.", prompt)
        self.assertIn("Chunk two.", prompt)
        self.assertIn("What is it?", prompt)

    def test_create_chat_prompt_with_system_and_messages(self):
        prompt = self.llm.create_chat_prompt(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            system_message="You are helpful.",
        )
        self.assertIn("You are helpful.", prompt)
        self.assertIn("Hi", prompt)
        self.assertIn("Hello!", prompt)

    def test_estimate_tokens_when_model_none_uses_word_estimate(self):
        self.llm.model = None
        n = self.llm.estimate_tokens("one two three four")
        self.assertGreater(n, 0)
        self.assertLessEqual(n, 10)

    def test_estimate_tokens_uses_model_when_loaded(self):
        self.llm.model.tokenize.return_value = [1] * 10
        self.assertEqual(self.llm.estimate_tokens("text"), 10)

    def test_truncate_to_context_when_model_none_uses_word_based(self):
        self.llm.model = None
        long_text = " ".join(["word"] * 200)
        out = self.llm.truncate_to_context(long_text, max_tokens=50, preserve_end=True)
        self.assertLessEqual(len(out.split()), 50 * 2)

    def test_get_model_info_when_not_loaded(self):
        self.llm.model = None
        info = self.llm.get_model_info()
        self.assertEqual(info["status"], "not_loaded")

    def test_get_model_info_when_loaded(self):
        info = self.llm.get_model_info()
        self.assertEqual(info["status"], "loaded")
        self.assertIn("context_length", info)

    def test_is_loaded(self):
        self.assertTrue(self.llm.is_loaded())
        self.llm.model = None
        self.assertFalse(self.llm.is_loaded())

    def test_unload_model_clears_model(self):
        self.llm.unload_model()
        self.assertIsNone(self.llm.model)

    def test_reload_model_calls_unload_and_load(self):
        with patch.object(self.llm, "_load_model") as load_mock:
            with patch.object(self.llm, "unload_model"):
                self.llm.reload_model()
        load_mock.assert_called_once()


class TestLLMManager(unittest.TestCase):
    def test_load_model_adds_and_sets_current(self):
        from offline_ai_assistant.llm.local_llm import LLMManager, LocalLLM

        path = MagicMock(spec=Path)
        path.exists.return_value = True
        path.stat.return_value = MagicMock(st_size=1)
        path.__str__ = lambda _: "/tmp/m.gguf"
        with patch.object(LocalLLM, "_load_model"):
            with patch("offline_ai_assistant.llm.local_llm.Llama", MagicMock()):
                with patch("offline_ai_assistant.llm.local_llm.Config") as cfg:
                    cfg.LLM_MODEL_PATH = Path("/tmp/fake.gguf")
                    cfg.LLM_CONTEXT_LENGTH = 2048
                    cfg.LLM_N_GPU_LAYERS = 0
                    cfg.LLM_N_BATCH = 512
                    cfg.LLM_N_THREADS = 2
                    cfg.PROMPT_TEMPLATES = {}
                    mgr = LLMManager()
                    llm = mgr.load_model("default", path)
                    self.assertIn("default", mgr.models)
                    self.assertEqual(mgr.current_model, "default")
                    self.assertIs(mgr.get_model(), llm)

    def test_set_current_model(self):
        from offline_ai_assistant.llm.local_llm import LLMManager

        mgr = LLMManager()
        mgr.models = {"a": MagicMock(), "b": MagicMock()}
        mgr.current_model = "a"
        self.assertTrue(mgr.set_current_model("b"))
        self.assertEqual(mgr.current_model, "b")
        self.assertFalse(mgr.set_current_model("c"))

    def test_list_models(self):
        from offline_ai_assistant.llm.local_llm import LLMManager

        mgr = LLMManager()
        mgr.models = {"a": 1, "b": 2}
        self.assertEqual(sorted(mgr.list_models()), ["a", "b"])

    def test_unload_model_removes_and_clears_current(self):
        from offline_ai_assistant.llm.local_llm import LLMManager

        mgr = LLMManager()
        m = MagicMock()
        mgr.models = {"only": m}
        mgr.current_model = "only"
        result = mgr.unload_model("only")
        self.assertTrue(result)
        m.unload_model.assert_called_once()
        self.assertNotIn("only", mgr.models)
        self.assertIsNone(mgr.current_model)

    def test_unload_all_clears_all_models(self):
        from offline_ai_assistant.llm.local_llm import LLMManager

        mgr = LLMManager()
        mgr.models = {"a": MagicMock(), "b": MagicMock()}
        mgr.current_model = "a"
        mgr.unload_all()
        self.assertEqual(len(mgr.models), 0)
        self.assertIsNone(mgr.current_model)


class TestCreateLLM(unittest.TestCase):
    def test_create_llm_returns_local_llm(self):
        from offline_ai_assistant.llm.local_llm import LocalLLM, create_llm

        path = MagicMock(spec=Path)
        path.exists.return_value = True
        path.stat.return_value = MagicMock(st_size=1)
        path.__str__ = lambda _: "/tmp/m.gguf"
        with patch.object(LocalLLM, "_load_model"):
            with patch("offline_ai_assistant.llm.local_llm.Llama", MagicMock()):
                with patch("offline_ai_assistant.llm.local_llm.Config") as cfg:
                    cfg.LLM_MODEL_PATH = Path("/tmp/fake.gguf")
                    cfg.LLM_CONTEXT_LENGTH = 2048
                    cfg.LLM_N_GPU_LAYERS = 0
                    cfg.LLM_N_BATCH = 512
                    cfg.LLM_N_THREADS = 2
                    cfg.PROMPT_TEMPLATES = {}
                    llm = create_llm(model_path=path)
                    self.assertIsInstance(llm, LocalLLM)


if __name__ == "__main__":
    unittest.main()
