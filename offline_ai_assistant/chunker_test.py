"""
Tests for offline_ai_assistant.chunker (compatibility shim).

Verifies that the package-root chunker module re-exports the expected
symbols from data.chunker and core.models.
"""

import unittest


class TestChunkerShim(unittest.TestCase):
    """Test that the root chunker module exposes the same API as data.chunker and core.models."""

    def test_text_chunker_importable(self):
        from offline_ai_assistant.chunker import TextChunker
        from offline_ai_assistant.data.chunker import TextChunker as DataTextChunker
        self.assertIs(TextChunker, DataTextChunker)

    def test_chunk_document_importable(self):
        from offline_ai_assistant.chunker import chunk_document
        from offline_ai_assistant.data.chunker import chunk_document as data_chunk_document
        self.assertIs(chunk_document, data_chunk_document)

    def test_text_chunk_importable(self):
        from offline_ai_assistant.chunker import TextChunk
        from offline_ai_assistant.core.models import TextChunk as CoreTextChunk
        self.assertIs(TextChunk, CoreTextChunk)

    def test_all_exports(self):
        import offline_ai_assistant.chunker as chunker_module
        self.assertIn("TextChunker", chunker_module.__all__)
        self.assertIn("chunk_document", chunker_module.__all__)
        self.assertIn("TextChunk", chunker_module.__all__)
        self.assertEqual(len(chunker_module.__all__), 3)


if __name__ == "__main__":
    unittest.main()
