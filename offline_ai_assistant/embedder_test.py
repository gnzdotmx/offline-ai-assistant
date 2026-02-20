"""
Tests for offline_ai_assistant.embedder (compatibility shim).

Verifies that the package-root embedder module re-exports the expected
symbols from data.embedder.
"""

import unittest


class TestEmbedderShim(unittest.TestCase):
    """Test that the root embedder module exposes the same API as data.embedder."""

    def test_text_embedder_importable(self):
        from offline_ai_assistant.embedder import TextEmbedder
        from offline_ai_assistant.data.embedder import TextEmbedder as DataTextEmbedder
        self.assertIs(TextEmbedder, DataTextEmbedder)

    def test_embedding_cache_importable(self):
        from offline_ai_assistant.embedder import EmbeddingCache
        from offline_ai_assistant.data.embedder import EmbeddingCache as DataEmbeddingCache
        self.assertIs(EmbeddingCache, DataEmbeddingCache)

    def test_create_embedder_importable(self):
        from offline_ai_assistant.embedder import create_embedder
        from offline_ai_assistant.data.embedder import create_embedder as data_create_embedder
        self.assertIs(create_embedder, data_create_embedder)

    def test_all_exports(self):
        import offline_ai_assistant.embedder as embedder_module
        self.assertIn("TextEmbedder", embedder_module.__all__)
        self.assertIn("EmbeddingCache", embedder_module.__all__)
        self.assertIn("create_embedder", embedder_module.__all__)
        self.assertEqual(len(embedder_module.__all__), 3)


if __name__ == "__main__":
    unittest.main()
