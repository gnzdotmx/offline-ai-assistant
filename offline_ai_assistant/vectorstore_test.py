"""
Tests for offline_ai_assistant.vectorstore (compatibility shim).

Verifies that the package-root vectorstore module re-exports the expected
symbols from data.vectorstore.
"""

import unittest


class TestVectorStoreShim(unittest.TestCase):
    """Test that the root vectorstore module exposes the same API as data.vectorstore."""

    def test_vector_store_importable(self):
        from offline_ai_assistant.vectorstore import VectorStore
        from offline_ai_assistant.data.vectorstore import VectorStore as DataVectorStore
        self.assertIs(VectorStore, DataVectorStore)

    def test_create_vector_store_importable(self):
        from offline_ai_assistant.vectorstore import create_vector_store
        from offline_ai_assistant.data.vectorstore import create_vector_store as data_create_vector_store
        self.assertIs(create_vector_store, data_create_vector_store)

    def test_all_exports(self):
        import offline_ai_assistant.vectorstore as vectorstore_module
        self.assertIn("VectorStore", vectorstore_module.__all__)
        self.assertIn("create_vector_store", vectorstore_module.__all__)
        self.assertEqual(len(vectorstore_module.__all__), 2)


if __name__ == "__main__":
    unittest.main()
