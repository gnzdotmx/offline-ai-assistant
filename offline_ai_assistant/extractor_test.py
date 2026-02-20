"""
Tests for offline_ai_assistant.extractor (compatibility shim).

Verifies that the package-root extractor module re-exports the expected
symbols from data.extractor.
"""

import unittest


class TestExtractorShim(unittest.TestCase):
    """Test that the root extractor module exposes the same API as data.extractor."""

    def test_document_extractor_importable(self):
        from offline_ai_assistant.extractor import DocumentExtractor
        from offline_ai_assistant.data.extractor import DocumentExtractor as DataDocumentExtractor
        self.assertIs(DocumentExtractor, DataDocumentExtractor)

    def test_extract_document_importable(self):
        from offline_ai_assistant.extractor import extract_document
        from offline_ai_assistant.data.extractor import extract_document as data_extract_document
        self.assertIs(extract_document, data_extract_document)

    def test_all_exports(self):
        import offline_ai_assistant.extractor as extractor_module
        self.assertIn("DocumentExtractor", extractor_module.__all__)
        self.assertIn("extract_document", extractor_module.__all__)
        self.assertEqual(len(extractor_module.__all__), 2)


if __name__ == "__main__":
    unittest.main()
