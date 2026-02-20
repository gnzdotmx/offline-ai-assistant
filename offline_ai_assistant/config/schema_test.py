"""
Tests for offline_ai_assistant.config.schema.

Covers validate_settings (clamping, types, string validation) and get_default_settings.
"""

import unittest

from offline_ai_assistant.config.schema import (
    CONFIG_BOUNDS,
    validate_settings,
    get_default_settings,
)


class TestValidateSettings(unittest.TestCase):
    def test_returns_tuple_of_dict_and_list(self):
        validated, warnings = validate_settings({})
        self.assertIsInstance(validated, dict)
        self.assertIsInstance(warnings, list)

    def test_empty_input_returns_empty_validated(self):
        validated, _ = validate_settings({})
        self.assertEqual(len(validated), 0)

    def test_valid_values_accepted(self):
        settings = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k_retrieval": 5,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_temperature": 0.3,
        }
        validated, warnings = validate_settings(settings)
        self.assertEqual(validated["chunk_size"], 512)
        self.assertEqual(validated["chunk_overlap"], 50)
        self.assertEqual(validated["top_k_retrieval"], 5)
        self.assertEqual(validated["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(validated["llm_temperature"], 0.3)
        self.assertEqual(len(warnings), 0)

    def test_chunk_size_clamped_to_bounds(self):
        validated, warnings = validate_settings({"chunk_size": 10})
        self.assertEqual(validated["chunk_size"], 64)
        self.assertTrue(any("chunk_size" in w for w in warnings))

        validated, warnings = validate_settings({"chunk_size": 10000})
        self.assertEqual(validated["chunk_size"], 4096)

    def test_chunk_overlap_clamped(self):
        validated, _ = validate_settings({"chunk_overlap": -1})
        self.assertEqual(validated["chunk_overlap"], 0)
        validated, _ = validate_settings({"chunk_overlap": 2000})
        self.assertEqual(validated["chunk_overlap"], 1024)

    def test_top_k_retrieval_clamped(self):
        validated, _ = validate_settings({"top_k_retrieval": 0})
        self.assertEqual(validated["top_k_retrieval"], 1)
        validated, _ = validate_settings({"top_k_retrieval": 100})
        self.assertEqual(validated["top_k_retrieval"], 50)

    def test_min_score_retrieval_clamped(self):
        validated, _ = validate_settings({"min_score_retrieval": -2.0})
        self.assertEqual(validated["min_score_retrieval"], -1.0)
        validated, _ = validate_settings({"min_score_retrieval": 1.5})
        self.assertEqual(validated["min_score_retrieval"], 1.0)

    def test_invalid_chunk_size_produces_warning_no_key(self):
        validated, warnings = validate_settings({"chunk_size": "not_a_number"})
        self.assertNotIn("chunk_size", validated)
        self.assertTrue(any("chunk_size" in w.lower() for w in warnings))

    def test_llm_model_path_stripped(self):
        validated, _ = validate_settings({"llm_model_path": "  /path/to/model.gguf  "})
        self.assertEqual(validated["llm_model_path"], "/path/to/model.gguf")

    def test_llm_model_path_empty_string_not_accepted(self):
        validated, _ = validate_settings({"llm_model_path": "   "})
        self.assertNotIn("llm_model_path", validated)

    def test_embedding_model_stripped_and_accepted(self):
        validated, _ = validate_settings({"embedding_model": "  all-mpnet-base-v2  "})
        self.assertEqual(validated["embedding_model"], "all-mpnet-base-v2")

    def test_rag_context_order_valid_values(self):
        validated, _ = validate_settings({"rag_context_order": "score"})
        self.assertEqual(validated["rag_context_order"], "score")
        validated, _ = validate_settings({"rag_context_order": "DOCUMENT_ORDER"})
        self.assertEqual(validated["rag_context_order"], "document_order")

    def test_rag_context_order_invalid_produces_warning(self):
        validated, warnings = validate_settings({"rag_context_order": "invalid"})
        self.assertNotIn("rag_context_order", validated)
        self.assertTrue(any("rag_context_order" in w.lower() for w in warnings))

    def test_rag_rerank_bool(self):
        validated, _ = validate_settings({"rag_rerank": True})
        self.assertTrue(validated["rag_rerank"])
        validated, _ = validate_settings({"rag_rerank": 0})
        self.assertFalse(validated["rag_rerank"])

    def test_embedding_batch_size_clamped(self):
        validated, _ = validate_settings({"embedding_batch_size": 0})
        self.assertEqual(validated["embedding_batch_size"], 1)
        validated, _ = validate_settings({"embedding_batch_size": 1000})
        self.assertEqual(validated["embedding_batch_size"], 512)

    def test_embedding_show_progress_bool(self):
        validated, _ = validate_settings({"embedding_show_progress": False})
        self.assertFalse(validated["embedding_show_progress"])

    def test_extractor_clean_text_bool(self):
        validated, _ = validate_settings({"extractor_clean_text": True})
        self.assertTrue(validated["extractor_clean_text"])

    def test_word_fallback_chunk_ratio_clamped(self):
        validated, _ = validate_settings({"word_fallback_chunk_ratio": 0.1})
        self.assertEqual(validated["word_fallback_chunk_ratio"], 0.25)
        validated, _ = validate_settings({"word_fallback_chunk_ratio": 1.5})
        self.assertEqual(validated["word_fallback_chunk_ratio"], 1.0)

    def test_encoding_model_accepted(self):
        validated, _ = validate_settings({"encoding_model": "cl100k_base"})
        self.assertEqual(validated["encoding_model"], "cl100k_base")

    def test_unknown_keys_ignored(self):
        validated, _ = validate_settings({"unknown_key": 123, "chunk_size": 256})
        self.assertNotIn("unknown_key", validated)
        self.assertEqual(validated["chunk_size"], 256)

    def test_llm_n_gpu_layers_clamped(self):
        validated, _ = validate_settings({"llm_n_gpu_layers": -2})
        self.assertEqual(validated["llm_n_gpu_layers"], -1)
        validated, _ = validate_settings({"llm_n_gpu_layers": 2000})
        self.assertEqual(validated["llm_n_gpu_layers"], 1024)

    def test_invalid_min_score_retrieval_produces_warning(self):
        validated, warnings = validate_settings({"min_score_retrieval": "not_a_float"})
        self.assertNotIn("min_score_retrieval", validated)
        self.assertTrue(any("min_score" in w.lower() for w in warnings))

    def test_invalid_top_k_retrieval_produces_warning(self):
        validated, warnings = validate_settings({"top_k_retrieval": "nope"})
        self.assertNotIn("top_k_retrieval", validated)
        self.assertTrue(any("top_k" in w.lower() for w in warnings))

    def test_rag_rerank_candidate_multiplier_clamped(self):
        validated, _ = validate_settings({"rag_rerank_candidate_multiplier": 1})
        self.assertEqual(validated["rag_rerank_candidate_multiplier"], 2)
        validated, _ = validate_settings({"rag_rerank_candidate_multiplier": 10})
        self.assertEqual(validated["rag_rerank_candidate_multiplier"], 5)

    def test_rag_max_chunks_per_doc_clamped(self):
        validated, _ = validate_settings({"rag_max_chunks_per_doc": -1})
        self.assertEqual(validated["rag_max_chunks_per_doc"], 0)
        validated, _ = validate_settings({"rag_max_chunks_per_doc": 100})
        self.assertEqual(validated["rag_max_chunks_per_doc"], 50)


class TestGetDefaultSettings(unittest.TestCase):
    def test_returns_dict(self):
        defaults = get_default_settings()
        self.assertIsInstance(defaults, dict)

    def test_contains_expected_keys(self):
        defaults = get_default_settings()
        expected = {
            "chunk_size",
            "chunk_overlap",
            "top_k_retrieval",
            "embedding_model",
            "llm_max_tokens",
            "llm_temperature",
            "embedding_batch_size",
            "rag_context_order",
            "extractor_clean_text",
        }
        for key in expected:
            self.assertIn(key, defaults, f"Missing key: {key}")

    def test_chunk_defaults(self):
        defaults = get_default_settings()
        self.assertEqual(defaults["chunk_size"], 512)
        self.assertEqual(defaults["chunk_overlap"], 50)

    def test_embedding_defaults(self):
        defaults = get_default_settings()
        self.assertEqual(defaults["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(defaults["embedding_batch_size"], 32)
        self.assertTrue(defaults["embedding_show_progress"])

    def test_llm_defaults(self):
        defaults = get_default_settings()
        self.assertEqual(defaults["llm_max_tokens"], 256)
        self.assertEqual(defaults["llm_temperature"], 0.3)
        self.assertEqual(defaults["llm_top_p"], 0.9)
        self.assertEqual(defaults["llm_n_gpu_layers"], 0)
        self.assertEqual(defaults["llm_n_batch"], 512)

    def test_rag_defaults(self):
        defaults = get_default_settings()
        self.assertEqual(defaults["rag_context_order"], "document_order")
        self.assertFalse(defaults["rag_rerank"])
        self.assertEqual(defaults["rag_max_chunks_per_doc"], 0)


class TestConfigBounds(unittest.TestCase):
    def test_bounds_defined_for_numeric_settings(self):
        self.assertIn("chunk_size", CONFIG_BOUNDS)
        self.assertIn("top_k_retrieval", CONFIG_BOUNDS)
        self.assertIn("embedding_batch_size", CONFIG_BOUNDS)

    def test_bounds_are_two_tuples(self):
        for key, bounds in CONFIG_BOUNDS.items():
            self.assertIsInstance(bounds, (list, tuple), key)
            self.assertEqual(len(bounds), 2, key)
            self.assertLessEqual(bounds[0], bounds[1], key)


if __name__ == "__main__":
    unittest.main()
