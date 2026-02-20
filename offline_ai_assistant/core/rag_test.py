"""
Tests for offline_ai_assistant.core.rag.

Uses mocks for embedder, vector_store, extractor, chunker, and LLM to avoid
real models and disk I/O. Pure helpers (_cap_chunks_per_document,
_reorder_results_by_document) are tested in isolation.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from offline_ai_assistant.core.rag import (
    _cap_chunks_per_document,
    _reorder_results_by_document,
    _apply_retrieval,
    RAGPipeline,
    create_rag_pipeline,
)
from offline_ai_assistant.core.models import (
    ProcessingResult,
    RAGResult,
    GenerationConfig,
    TextChunk,
)


class TestCapChunksPerDocument(unittest.TestCase):
    def test_returns_all_when_max_per_doc_zero(self):
        results = [
            {"document_id": 1, "text": "a"},
            {"document_id": 1, "text": "b"},
        ]
        out = _cap_chunks_per_document(results, top_k=10, max_per_doc=0)
        self.assertEqual(out, results)

    def test_returns_all_when_empty(self):
        out = _cap_chunks_per_document([], top_k=5, max_per_doc=2)
        self.assertEqual(out, [])

    def test_caps_per_document_fills_with_others(self):
        # doc 1: 3 chunks, doc 2: 2 chunks; max_per_doc=2, top_k=4
        results = [
            {"document_id": 1, "text": "1a"},
            {"document_id": 1, "text": "1b"},
            {"document_id": 1, "text": "1c"},
            {"document_id": 2, "text": "2a"},
            {"document_id": 2, "text": "2b"},
        ]
        out = _cap_chunks_per_document(results, top_k=4, max_per_doc=2)
        self.assertEqual(len(out), 4)
        doc1_count = sum(1 for r in out if r["document_id"] == 1)
        doc2_count = sum(1 for r in out if r["document_id"] == 2)
        self.assertLessEqual(doc1_count, 2)
        self.assertLessEqual(doc2_count, 2)

    def test_stops_at_top_k(self):
        results = [
            {"document_id": 1, "text": "a"},
            {"document_id": 2, "text": "b"},
            {"document_id": 3, "text": "c"},
        ]
        out = _cap_chunks_per_document(results, top_k=2, max_per_doc=1)
        self.assertEqual(len(out), 2)

    def test_none_document_id_treated_as_zero(self):
        # Both items end up as doc_id 0; with max_per_doc=1 only one is kept
        results = [
            {"document_id": None, "text": "a"},
            {"text": "b"},
        ]
        out = _cap_chunks_per_document(results, top_k=5, max_per_doc=1)
        self.assertEqual(len(out), 1)


class TestReorderResultsByDocument(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(_reorder_results_by_document([]), [])

    def test_single_doc_preserves_order_by_chunk_index(self):
        results = [
            {"document_id": 1, "chunk_index": 2, "rank": 1},
            {"document_id": 1, "chunk_index": 0, "rank": 2},
            {"document_id": 1, "chunk_index": 1, "rank": 3},
        ]
        out = _reorder_results_by_document(results)
        self.assertEqual([r["chunk_index"] for r in out], [0, 1, 2])

    def test_multi_doc_orders_groups_by_best_rank(self):
        # Doc 2 has best rank 1 (one chunk), doc 1 has ranks 2,3 (two chunks)
        results = [
            {"document_id": 1, "chunk_index": 0, "rank": 2},
            {"document_id": 1, "chunk_index": 1, "rank": 3},
            {"document_id": 2, "chunk_index": 0, "rank": 1},
        ]
        out = _reorder_results_by_document(results)
        self.assertEqual(out[0]["document_id"], 2)
        self.assertEqual(out[1]["document_id"], 1)
        self.assertEqual(out[2]["document_id"], 1)


class TestApplyRetrieval(unittest.TestCase):
    def test_returns_search_results_with_rank_when_no_rerank_no_cap(self):
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.RAG_RERANK = False
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            mock_vs = MagicMock()
            mock_vs.search.return_value = [
                {"document_id": 1, "text": "a", "score": 0.9, "file_name": "f", "chunk_index": 0,
                 "file_path": "/f", "start_char": 0, "end_char": 10},
            ]
            out = _apply_retrieval("q", "embedding", mock_vs, effective_top_k=5, min_score=0.0)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["rank"], 1)
            mock_vs.search.assert_called_once()

    def test_rerank_path_calls_rerank_and_caps_candidates(self):
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.RAG_RERANK = True
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            cfg.RAG_RERANK_CANDIDATE_MULTIPLIER = 3
            mock_vs = MagicMock()
            mock_vs.search.return_value = [
                {"document_id": 1, "text": "a", "score": 0.9, "file_name": "f", "chunk_index": 0,
                 "file_path": "/f", "start_char": 0, "end_char": 10},
            ]
            with patch("offline_ai_assistant.core.rag.rerank_module.rerank") as mock_rerank:
                mock_rerank.return_value = mock_vs.search.return_value[:1]
                out = _apply_retrieval("q", "embedding", mock_vs, effective_top_k=5, min_score=0.0)
            self.assertEqual(len(out), 1)
            mock_rerank.assert_called_once()
            self.assertEqual(mock_vs.search.call_args[1]["top_k"], min(15, 50))

    def test_max_chunks_per_doc_caps_results(self):
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.RAG_RERANK = False
            cfg.RAG_MAX_CHUNKS_PER_DOC = 1
            mock_vs = MagicMock()
            mock_vs.search.return_value = [
                {"document_id": 1, "text": "a", "score": 0.9, "file_name": "f", "chunk_index": 0,
                 "file_path": "/f", "start_char": 0, "end_char": 10},
                {"document_id": 1, "text": "b", "score": 0.8, "file_name": "f", "chunk_index": 1,
                 "file_path": "/f", "start_char": 10, "end_char": 20},
                {"document_id": 2, "text": "c", "score": 0.7, "file_name": "g", "chunk_index": 0,
                 "file_path": "/g", "start_char": 0, "end_char": 10},
            ]
            out = _apply_retrieval("q", "embedding", mock_vs, effective_top_k=3, min_score=0.0)
            self.assertEqual(len(out), 2)
            doc1_count = sum(1 for r in out if r.get("document_id") == 1)
            self.assertLessEqual(doc1_count, 1)

    def test_rerank_score_copied_to_score_when_rerank_enabled(self):
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.RAG_RERANK = True
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            cfg.RAG_RERANK_CANDIDATE_MULTIPLIER = 2
            mock_vs = MagicMock()
            mock_vs.search.return_value = [
                {"document_id": 1, "text": "a", "score": 0.5, "rerank_score": 0.95, "file_name": "f",
                 "chunk_index": 0, "file_path": "/f", "start_char": 0, "end_char": 10},
            ]
            with patch("offline_ai_assistant.core.rag.rerank_module.rerank") as mock_rerank:
                mock_rerank.return_value = mock_vs.search.return_value
                out = _apply_retrieval("q", "embedding", mock_vs, effective_top_k=5, min_score=0.0)
            self.assertEqual(out[0]["score"], 0.95)


class TestRAGPipelineInit(unittest.TestCase):
    def test_init_with_mocks(self):
        embedder = MagicMock()
        embedder.embedding_dim = 384
        vs = MagicMock()
        pipeline = RAGPipeline(embedder=embedder, vector_store=vs)
        self.assertIs(pipeline.embedder, embedder)
        self.assertIs(pipeline.vector_store, vs)
        self.assertIsNone(pipeline.llm)
        self.assertEqual(pipeline.stats["documents_processed"], 0)
        self.assertEqual(pipeline.stats["queries_answered"], 0)


class TestRAGPipelineListDocuments(unittest.TestCase):
    def test_list_documents_delegates_to_vector_store(self):
        vs = MagicMock()
        vs.list_documents.return_value = [{"document_id": 1, "file_name": "doc.pdf"}]
        pipeline = RAGPipeline(vector_store=vs)
        result = pipeline.list_documents()
        self.assertEqual(result, [{"document_id": 1, "file_name": "doc.pdf"}])
        vs.list_documents.assert_called_once()


class TestRAGPipelineDeleteDocument(unittest.TestCase):
    def test_delete_document_delegates_and_returns_success(self):
        vs = MagicMock()
        vs.list_documents.return_value = [
            {"document_id": 1, "file_path": "/nonexistent/doc.pdf"},
        ]
        vs.delete_document.return_value = True
        pipeline = RAGPipeline(vector_store=vs)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.DOCS_DIR = Path("/tmp/docs")
            result = pipeline.delete_document(1)
        self.assertTrue(result)
        vs.delete_document.assert_called_once_with(1)


class TestRAGPipelineGetDocumentChunks(unittest.TestCase):
    def test_get_document_chunks_delegates(self):
        vs = MagicMock()
        vs.get_document_chunks.return_value = [{"chunk_index": 0, "text": "chunk"}]
        pipeline = RAGPipeline(vector_store=vs)
        result = pipeline.get_document_chunks(1)
        self.assertEqual(result, [{"chunk_index": 0, "text": "chunk"}])
        vs.get_document_chunks.assert_called_once_with(1)


class TestRAGPipelineGetStatistics(unittest.TestCase):
    def test_get_statistics_returns_aggregate(self):
        vs = MagicMock()
        vs.get_stats.return_value = {"documents": 2, "chunks": 10}
        embedder = MagicMock()
        embedder.get_model_info.return_value = {"embedding_dim": 384}
        pipeline = RAGPipeline(vector_store=vs, embedder=embedder)
        result = pipeline.get_statistics()
        self.assertIn("pipeline_stats", result)
        self.assertIn("vector_store", result)
        self.assertIn("embedder", result)
        self.assertEqual(result["vector_store"]["documents"], 2)


class TestRAGPipelineClose(unittest.TestCase):
    def test_close_closes_vector_store_and_unloads_llm(self):
        vs = MagicMock()
        llm = MagicMock()
        pipeline = RAGPipeline(vector_store=vs, llm=llm)
        pipeline.close()
        vs.close.assert_called_once()
        llm.unload_model.assert_called_once()


class TestRAGPipelineProcessDocument(unittest.TestCase):
    def test_process_document_invalid_file_returns_failure(self):
        extractor = MagicMock()
        extractor.validate_file.return_value = (False, "Not a PDF")
        pipeline = RAGPipeline(extractor=extractor)
        result = pipeline.process_document(Path("/tmp/invalid.pdf"))
        self.assertIsInstance(result, ProcessingResult)
        self.assertFalse(result.success)
        self.assertIsNone(result.document_id)
        self.assertEqual(result.chunks_created, 0)
        self.assertIn("Not a PDF", result.error_message or "")

    def test_process_document_duplicate_returns_early_success(self):
        extractor = MagicMock()
        extractor.validate_file.return_value = (True, "")
        extractor.extract_from_file.return_value = {"file_hash": "abc123", "full_text": "x"}
        vs = MagicMock()
        vs.list_documents.return_value = [
            {"document_id": 42, "file_hash": "abc123", "chunk_count": 5},
        ]
        pipeline = RAGPipeline(extractor=extractor, vector_store=vs)
        result = pipeline.process_document(Path("/tmp/doc.pdf"))
        self.assertTrue(result.success)
        self.assertEqual(result.document_id, 42)
        self.assertEqual(result.chunks_created, 5)
        vs.add_document.assert_not_called()

    def test_process_document_success_with_mocks(self):
        extractor = MagicMock()
        extractor.validate_file.return_value = (True, "")
        extractor.extract_from_file.return_value = {
            "file_hash": "new123",
            "full_text": "Some text here.",
            "file_name": "doc.pdf",
            "file_type": "application/pdf",
            "file_size": 100,
            "extraction_date": "2024-01-01",
            "metadata": {},
        }
        vs = MagicMock()
        vs.list_documents.return_value = []
        vs.add_document.return_value = 1
        chunker = MagicMock()
        chunker.chunk_text.return_value = [
            TextChunk("Some text", 0, 9, 3, 0, "doc.pdf"),
        ]
        embedder = MagicMock()
        embedder.embed_chunks.return_value = [
            {
                "text": "Some text",
                "start_char": 0,
                "end_char": 9,
                "token_count": 3,
                "chunk_index": 0,
                "embedding": [0.1] * 384,
                "embedding_model": "test",
                "embedding_dim": 384,
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            try:
                path = Path(f.name)
                pipeline = RAGPipeline(
                    extractor=extractor,
                    vector_store=vs,
                    chunker=chunker,
                    embedder=embedder,
                )
                with patch.object(pipeline, "_copy_document_to_storage", return_value=path):
                    with patch("offline_ai_assistant.core.rag.Config") as cfg:
                        cfg.EMBEDDING_BATCH_SIZE = 32
                        cfg.EMBEDDING_SHOW_PROGRESS = False
                        result = pipeline.process_document(path)
                self.assertTrue(result.success)
                self.assertEqual(result.document_id, 1)
                self.assertEqual(result.chunks_created, 1)
                vs.add_document.assert_called_once()
            finally:
                path.unlink(missing_ok=True)


class TestRAGPipelineQuery(unittest.TestCase):
    def test_query_no_llm_raises(self):
        pipeline = RAGPipeline(llm=None)
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.query("test query")
        self.assertIn("LLM not initialized", str(ctx.exception))

    def test_query_no_search_results_returns_fallback(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        vs = MagicMock()
        vs.search.return_value = []
        llm = MagicMock()
        pipeline = RAGPipeline(embedder=embedder, vector_store=vs, llm=llm)
        result = pipeline.query("test query")
        self.assertIsInstance(result, RAGResult)
        self.assertIn("couldn't find any relevant", result.answer)
        self.assertEqual(result.sources, [])
        llm.generate_complete.assert_not_called()

    def test_query_with_results_returns_rag_result(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        vs = MagicMock()
        vs.search.return_value = [
            {
                "document_id": 1,
                "text": "Relevant context.",
                "score": 0.9,
                "rank": 1,
                "file_name": "doc.pdf",
                "file_path": "/path/doc.pdf",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 18,
            },
        ]
        llm = MagicMock()
        llm.model_path = MagicMock()
        llm.model_path.name = "test-model"
        llm.create_rag_prompt.return_value = "Prompt with context"
        llm.truncate_to_context.side_effect = lambda x, **kw: x
        llm.generate_complete.return_value = "Generated answer."
        llm.estimate_tokens.return_value = 5
        pipeline = RAGPipeline(embedder=embedder, vector_store=vs, llm=llm)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.TOP_K_RETRIEVAL = 5
            cfg.MIN_SCORE_RETRIEVAL = 0.0
            cfg.RAG_CONTEXT_ORDER = "score"
            cfg.RAG_RERANK = False
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            cfg.PROMPT_TEMPLATES = {"default": "Context:\n{context}\n\nQuestion: {question}"}
            cfg.LLM_MAX_TOKENS = 256
            cfg.LLM_TEMPERATURE = 0.3
            cfg.LLM_TOP_P = 0.9
            result = pipeline.query("test query")
        self.assertIsInstance(result, RAGResult)
        self.assertEqual(result.answer, "Generated answer.")
        self.assertEqual(len(result.sources), 1)
        self.assertGreater(result.total_time, 0)

    def test_query_uses_document_order_when_configured(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        vs = MagicMock()
        vs.search.return_value = [
            {"document_id": 1, "text": "First.", "score": 0.8, "rank": 2, "file_name": "a.pdf",
             "file_path": "/a.pdf", "chunk_index": 1, "start_char": 0, "end_char": 6},
            {"document_id": 1, "text": "Second.", "score": 0.9, "rank": 1, "file_name": "a.pdf",
             "file_path": "/a.pdf", "chunk_index": 0, "start_char": 0, "end_char": 7},
        ]
        llm = MagicMock()
        llm.model_path = MagicMock()
        llm.model_path.name = "test"
        llm.create_rag_prompt.return_value = "prompt"
        llm.truncate_to_context.side_effect = lambda x, **kw: x
        llm.generate_complete.return_value = "Answer."
        llm.estimate_tokens.return_value = 1
        pipeline = RAGPipeline(embedder=embedder, vector_store=vs, llm=llm)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.TOP_K_RETRIEVAL = 5
            cfg.MIN_SCORE_RETRIEVAL = 0.0
            cfg.RAG_CONTEXT_ORDER = "document_order"
            cfg.RAG_RERANK = False
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            cfg.PROMPT_TEMPLATES = {"default": "{context}\n\n{question}"}
            cfg.LLM_MAX_TOKENS = 256
            cfg.LLM_TEMPERATURE = 0.3
            cfg.LLM_TOP_P = 0.9
            result = pipeline.query("q")
        self.assertEqual(result.answer, "Answer.")
        self.assertEqual(len(result.sources), 2)
        self.assertEqual(result.sources[0]["chunk_index"], 0)
        self.assertEqual(result.sources[1]["chunk_index"], 1)


class TestRAGPipelineQueryStream(unittest.TestCase):
    def test_query_stream_no_llm_raises(self):
        pipeline = RAGPipeline(llm=None)
        with self.assertRaises(RuntimeError) as ctx:
            list(pipeline.query_stream("q"))
        self.assertIn("LLM not initialized", str(ctx.exception))

    def test_query_stream_llm_not_loaded_raises(self):
        llm = MagicMock()
        llm.is_loaded.return_value = False
        pipeline = RAGPipeline(llm=llm)
        with self.assertRaises(RuntimeError) as ctx:
            list(pipeline.query_stream("q"))
        self.assertIn("not loaded", str(ctx.exception))

    def test_query_stream_no_results_yields_final_and_returns(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        vs = MagicMock()
        vs.search.return_value = []
        llm = MagicMock()
        llm.is_loaded.return_value = True
        pipeline = RAGPipeline(embedder=embedder, vector_store=vs, llm=llm)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.TOP_K_RETRIEVAL = 5
            cfg.MIN_SCORE_RETRIEVAL = 0.0
            cfg.RAG_RERANK = False
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            cfg.RAG_CONTEXT_ORDER = "score"
            cfg.PROMPT_TEMPLATES = {"default": "{context}\n{question}"}
            updates = list(pipeline.query_stream("q"))
        types = [u["type"] for u in updates]
        self.assertIn("status", types)
        self.assertIn("final", types)
        final = next(u for u in updates if u["type"] == "final")
        self.assertIn("couldn't find any relevant", final["answer"])
        self.assertEqual(final["chunks_retrieved"], 0)

    def test_query_stream_with_results_yields_sources_tokens_final(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        vs = MagicMock()
        vs.search.return_value = [
            {"document_id": 1, "text": "ctx", "score": 0.9, "rank": 1, "file_name": "f",
             "file_path": "/f", "chunk_index": 0, "start_char": 0, "end_char": 3},
        ]
        llm = MagicMock()
        llm.is_loaded.return_value = True
        llm.n_ctx = 2048
        llm.create_rag_prompt.return_value = "prompt"
        llm.truncate_to_context.side_effect = lambda x, **kw: x
        llm.generate.return_value = iter(["Hello ", "world"])
        llm.estimate_tokens.return_value = 2
        pipeline = RAGPipeline(embedder=embedder, vector_store=vs, llm=llm)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.TOP_K_RETRIEVAL = 5
            cfg.MIN_SCORE_RETRIEVAL = 0.0
            cfg.RAG_RERANK = False
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            cfg.RAG_CONTEXT_ORDER = "score"
            cfg.PROMPT_TEMPLATES = {"default": "{context}\n{question}"}
            cfg.LLM_MAX_TOKENS = 256
            cfg.LLM_TEMPERATURE = 0.3
            cfg.LLM_TOP_P = 0.9
            updates = list(pipeline.query_stream("q"))
        types = [u["type"] for u in updates]
        self.assertIn("sources", types)
        self.assertIn("token", types)
        self.assertIn("final", types)
        tokens = [u for u in updates if u["type"] == "token"]
        self.assertEqual([u["token"] for u in tokens], ["Hello ", "world"])
        final = next(u for u in updates if u["type"] == "final")
        self.assertEqual(final["answer"], "Hello world")

    def test_query_stream_exception_yields_error(self):
        embedder = MagicMock()
        embedder.embed_query.side_effect = RuntimeError("embed failed")
        llm = MagicMock()
        llm.is_loaded.return_value = True
        pipeline = RAGPipeline(embedder=embedder, llm=llm)
        with patch("offline_ai_assistant.core.rag.Config"):
            updates = list(pipeline.query_stream("q"))
        err = next((u for u in updates if u["type"] == "error"), None)
        self.assertIsNotNone(err)
        self.assertIn("embed failed", err["error"])


class TestRAGPipelineGenerateContent(unittest.TestCase):
    def test_generate_content_no_llm_raises(self):
        pipeline = RAGPipeline(llm=None)
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.generate_content("default", direct_prompt="Say hi")
        self.assertIn("LLM not initialized", str(ctx.exception))

    def test_generate_content_neither_context_nor_direct_raises(self):
        llm = MagicMock()
        pipeline = RAGPipeline(llm=llm)
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.generate_content("default")
        self.assertIn("context_query or direct_prompt", str(ctx.exception))

    def test_generate_content_direct_prompt_returns_rag_result(self):
        llm = MagicMock()
        llm.model_path = MagicMock()
        llm.model_path.name = "m"
        llm.generate_complete.return_value = "Generated text"
        llm.estimate_tokens.return_value = 2
        pipeline = RAGPipeline(llm=llm)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.LLM_MAX_TOKENS = 256
            cfg.LLM_TEMPERATURE = 0.3
            cfg.LLM_TOP_P = 0.9
            result = pipeline.generate_content("default", direct_prompt="Hello")
        self.assertIsInstance(result, RAGResult)
        self.assertEqual(result.answer, "Generated text")
        self.assertEqual(result.sources, [])
        llm.generate_complete.assert_called_once()

    def test_generate_content_context_query_uses_retrieval(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        vs = MagicMock()
        vs.search.return_value = [
            {"document_id": 1, "text": "Context.", "score": 0.9, "rank": 1, "file_name": "f",
             "file_path": "/f", "chunk_index": 0, "start_char": 0, "end_char": 8},
        ]
        llm = MagicMock()
        llm.model_path = MagicMock()
        llm.model_path.name = "m"
        llm.generate_complete.return_value = "Answer from context"
        llm.estimate_tokens.return_value = 4
        pipeline = RAGPipeline(embedder=embedder, vector_store=vs, llm=llm)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.TOP_K_RETRIEVAL = 5
            cfg.MIN_SCORE_RETRIEVAL = 0.0
            cfg.RAG_RERANK = False
            cfg.RAG_MAX_CHUNKS_PER_DOC = 0
            cfg.RAG_CONTEXT_ORDER = "score"
            cfg.PROMPT_TEMPLATES = {"default": "Context:\n{context}\n\nQuestion: {question}"}
            cfg.LLM_MAX_TOKENS = 256
            cfg.LLM_TEMPERATURE = 0.3
            cfg.LLM_TOP_P = 0.9
            result = pipeline.generate_content("default", context_query="What is it?")
        self.assertEqual(result.answer, "Answer from context")
        self.assertEqual(len(result.sources), 1)
        self.assertGreater(result.retrieval_time, 0)


class TestRAGPipelineGetStatistics(unittest.TestCase):
    def test_get_statistics_without_llm_includes_not_loaded(self):
        vs = MagicMock()
        vs.get_stats.return_value = {"documents": 0, "chunks": 0}
        embedder = MagicMock()
        embedder.get_model_info.return_value = {"embedding_dim": 384}
        pipeline = RAGPipeline(vector_store=vs, embedder=embedder, llm=None)
        result = pipeline.get_statistics()
        self.assertEqual(result["llm"]["status"], "not_loaded")
        self.assertIn("avg_processing_time", result)
        self.assertIn("avg_query_time", result)


class TestRAGPipelineDeleteDocument(unittest.TestCase):
    def test_delete_document_success_without_file_in_docs_dir(self):
        vs = MagicMock()
        vs.list_documents.return_value = [
            {"document_id": 1, "file_path": "/other/location/doc.pdf"},
        ]
        vs.delete_document.return_value = True
        pipeline = RAGPipeline(vector_store=vs)
        with patch("offline_ai_assistant.core.rag.Config") as cfg:
            cfg.DOCS_DIR = Path("/tmp/docs")
            result = pipeline.delete_document(1)
        self.assertTrue(result)
        vs.delete_document.assert_called_once_with(1)

    def test_delete_document_exception_returns_false(self):
        vs = MagicMock()
        vs.list_documents.side_effect = RuntimeError("db error")
        pipeline = RAGPipeline(vector_store=vs)
        result = pipeline.delete_document(1)
        self.assertFalse(result)


class TestRAGPipelineCopyDocumentToStorage(unittest.TestCase):
    def test_copy_document_to_storage_copies_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "docs"
            docs_dir.mkdir()
            source = Path(tmp) / "original.pdf"
            source.write_text("content")
            pipeline = RAGPipeline()
            with patch("offline_ai_assistant.core.rag.Config") as cfg:
                cfg.DOCS_DIR = docs_dir
                dest = pipeline._copy_document_to_storage(source)
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_text(), "content")
            self.assertEqual(dest.parent, docs_dir)

    def test_copy_document_to_storage_avoids_collision_with_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "docs"
            docs_dir.mkdir()
            existing = docs_dir / "doc.pdf"
            existing.write_text("existing")
            source = Path(tmp) / "doc.pdf"
            source.write_text("new")
            pipeline = RAGPipeline()
            with patch("offline_ai_assistant.core.rag.Config") as cfg:
                cfg.DOCS_DIR = docs_dir
                dest = pipeline._copy_document_to_storage(source)
            self.assertEqual(dest.name, "doc_1.pdf")
            self.assertEqual(dest.read_text(), "new")

    def test_copy_document_to_storage_on_copy_failure_returns_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "docs"
            docs_dir.mkdir()
            source = Path(tmp) / "readonly.pdf"
            source.write_text("content")
            pipeline = RAGPipeline()
            with patch("offline_ai_assistant.core.rag.Config") as cfg:
                cfg.DOCS_DIR = docs_dir
                with patch("offline_ai_assistant.core.rag.shutil.copy2", side_effect=OSError("permission")):
                    dest = pipeline._copy_document_to_storage(source)
            self.assertEqual(dest, source)


class TestCreateRAGPipeline(unittest.TestCase):
    def test_create_rag_pipeline_returns_rag_pipeline(self):
        with patch("offline_ai_assistant.core.rag.TextEmbedder") as mock_embedder_cls:
            with patch("offline_ai_assistant.core.rag.VectorStore") as mock_vs_cls:
                mock_embedder = MagicMock()
                mock_embedder.embedding_dim = 384
                mock_embedder_cls.return_value = mock_embedder
                mock_vs = MagicMock()
                mock_vs_cls.return_value = mock_vs
                result = create_rag_pipeline(model_path=None, embedding_model="all-MiniLM-L6-v2")
                self.assertIsInstance(result, RAGPipeline)
                self.assertIs(result.embedder, mock_embedder)
                self.assertIs(result.vector_store, mock_vs)
                self.assertIsNone(result.llm)

    def test_create_rag_pipeline_with_nonexistent_model_path_skips_llm(self):
        with patch("offline_ai_assistant.core.rag.TextEmbedder") as mock_embedder_cls:
            with patch("offline_ai_assistant.core.rag.VectorStore") as mock_vs_cls:
                mock_embedder = MagicMock()
                mock_embedder.embedding_dim = 384
                mock_embedder_cls.return_value = mock_embedder
                mock_vs_cls.return_value = MagicMock()
                result = create_rag_pipeline(
                    model_path=Path("/nonexistent/model.gguf"),
                    embedding_model="all-MiniLM-L6-v2",
                )
                self.assertIsInstance(result, RAGPipeline)
                self.assertIs(result.embedder, mock_embedder)
                self.assertIsNone(result.llm)


if __name__ == "__main__":
    unittest.main()
