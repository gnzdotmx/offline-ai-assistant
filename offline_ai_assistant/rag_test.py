"""
Tests for offline_ai_assistant.rag (compatibility shim).

Verifies that the package-root rag module re-exports the expected
symbols from core.rag and core.models.
"""

import unittest


class TestRAGShim(unittest.TestCase):
    """Test that the root rag module exposes the same API as core.rag and core.models."""

    def test_rag_pipeline_importable(self):
        from offline_ai_assistant.rag import RAGPipeline
        from offline_ai_assistant.core.rag import RAGPipeline as CoreRAGPipeline
        self.assertIs(RAGPipeline, CoreRAGPipeline)

    def test_create_rag_pipeline_importable(self):
        from offline_ai_assistant.rag import create_rag_pipeline
        from offline_ai_assistant.core.rag import create_rag_pipeline as core_create_rag_pipeline
        self.assertIs(create_rag_pipeline, core_create_rag_pipeline)

    def test_rag_result_importable(self):
        from offline_ai_assistant.rag import RAGResult
        from offline_ai_assistant.core.models import RAGResult as CoreRAGResult
        self.assertIs(RAGResult, CoreRAGResult)

    def test_processing_result_importable(self):
        from offline_ai_assistant.rag import ProcessingResult
        from offline_ai_assistant.core.models import ProcessingResult as CoreProcessingResult
        self.assertIs(ProcessingResult, CoreProcessingResult)

    def test_all_exports(self):
        import offline_ai_assistant.rag as rag_module
        self.assertIn("RAGPipeline", rag_module.__all__)
        self.assertIn("create_rag_pipeline", rag_module.__all__)
        self.assertIn("RAGResult", rag_module.__all__)
        self.assertIn("ProcessingResult", rag_module.__all__)
        self.assertEqual(len(rag_module.__all__), 4)


if __name__ == "__main__":
    unittest.main()
