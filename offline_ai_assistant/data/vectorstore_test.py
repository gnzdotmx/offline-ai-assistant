"""
Tests for offline_ai_assistant.data.vectorstore.

Uses a temporary directory for index and database. Requires FAISS to be installed.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from offline_ai_assistant.data.vectorstore import (
    VectorStore,
    create_vector_store,
    DEFAULT_EMBEDDING_DIM,
)


def _make_chunks_data(num_chunks: int, dim: int = 4) -> list:
    """Return minimal chunks_data for add_document (embeddings normalized for FAISS inner product)."""
    chunks = []
    for i in range(num_chunks):
        emb = np.ones(dim, dtype=np.float32) * (i + 1)
        emb = emb / np.linalg.norm(emb)
        chunks.append({
            "text": f"chunk {i}",
            "start_char": i * 10,
            "end_char": i * 10 + 8,
            "token_count": 5,
            "chunk_index": i,
            "embedding": emb,
            "embedding_model": "test-model",
            "embedding_dim": dim,
        })
    return chunks


def _make_document_data(file_hash: str = "abc123") -> dict:
    return {
        "file_path": "/tmp/doc.pdf",
        "file_name": "doc.pdf",
        "file_type": "application/pdf",
        "file_size": 1000,
        "file_hash": file_hash,
        "extraction_date": "2024-01-01T00:00:00",
        "metadata": {},
    }


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreInit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp = Path(self.tmp)
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_init_creates_db_and_index(self):
        vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)
        self.assertTrue(self.db_path.exists())
        self.assertIsNotNone(vs.index)
        self.assertEqual(vs.embedding_dim, 4)
        vs.close()

    def test_init_without_embedding_dim_uses_default_when_empty(self):
        vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=None)
        self.assertEqual(vs.embedding_dim, DEFAULT_EMBEDDING_DIM)
        vs.close()


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreAddDocument(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_document_empty_chunks_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.vs.add_document(_make_document_data(), [])
        self.assertIn("No chunks", str(ctx.exception))

    def test_add_document_returns_document_id(self):
        doc_id = self.vs.add_document(
            _make_document_data(),
            _make_chunks_data(2, dim=4),
        )
        self.assertIsInstance(doc_id, int)
        self.assertEqual(doc_id, 1)

    def test_add_document_duplicate_hash_returns_existing_id(self):
        doc_data = _make_document_data("same")
        id1 = self.vs.add_document(doc_data, _make_chunks_data(1, dim=4))
        id2 = self.vs.add_document(doc_data, _make_chunks_data(1, dim=4))
        self.assertEqual(id1, id2)


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreListDocuments(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_documents_empty(self):
        self.assertEqual(self.vs.list_documents(), [])

    def test_list_documents_after_add(self):
        self.vs.add_document(_make_document_data("h1"), _make_chunks_data(2, dim=4))
        docs = self.vs.list_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["file_name"], "doc.pdf")
        self.assertEqual(docs[0]["chunk_count"], 2)


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_empty_index_returns_empty(self):
        q = np.ones(4, dtype=np.float32) / np.sqrt(4)
        self.assertEqual(self.vs.search(q, top_k=5), [])

    def test_search_returns_results_after_add(self):
        self.vs.add_document(_make_document_data("h1"), _make_chunks_data(3, dim=4))
        q = np.ones(4, dtype=np.float32) / np.sqrt(4)
        results = self.vs.search(q, top_k=5, min_score=0.0)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("chunk_id", r)
            self.assertIn("score", r)
            self.assertIn("text", r)
            self.assertIn("file_name", r)
            self.assertIn("document_id", r)


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreGetDocumentChunks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_document_chunks_returns_chunks(self):
        doc_id = self.vs.add_document(_make_document_data(), _make_chunks_data(2, dim=4))
        chunks = self.vs.get_document_chunks(doc_id)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "chunk 0")
        self.assertEqual(chunks[1]["chunk_index"], 1)


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreDeleteDocument(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_delete_document_removes_from_list(self):
        doc_id = self.vs.add_document(_make_document_data("del"), _make_chunks_data(1, dim=4))
        self.assertEqual(len(self.vs.list_documents()), 1)
        success = self.vs.delete_document(doc_id)
        self.assertTrue(success)
        self.assertEqual(len(self.vs.list_documents()), 0)

    def test_delete_document_nonexistent_returns_false(self):
        self.assertFalse(self.vs.delete_document(99999))


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreClearAllDocuments(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clear_all_documents_empties_store(self):
        self.vs.add_document(_make_document_data("a"), _make_chunks_data(1, dim=4))
        self.assertEqual(len(self.vs.list_documents()), 1)
        self.vs.clear_all_documents(new_embedding_dim=4)
        self.assertEqual(len(self.vs.list_documents()), 0)
        self.assertEqual(self.vs.index.ntotal, 0)


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreGetIndexEmbeddingModel(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_index_embedding_model_empty_returns_none(self):
        self.assertIsNone(self.vs.get_index_embedding_model())

    def test_get_index_embedding_model_after_add_returns_model(self):
        self.vs.add_document(_make_document_data(), _make_chunks_data(1, dim=4))
        self.assertEqual(self.vs.get_index_embedding_model(), "test-model")


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestVectorStoreGetStats(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.index_path = self.tmp / "index.faiss"
        self.db_path = self.tmp / "store.db"
        self.vs = VectorStore(index_path=self.index_path, db_path=self.db_path, embedding_dim=4)

    def tearDown(self):
        self.vs.close()
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_stats_returns_dict(self):
        stats = self.vs.get_stats()
        self.assertIn("documents", stats)
        self.assertIn("chunks", stats)
        self.assertIn("vectors_in_index", stats)
        self.assertIn("embedding_dim", stats)
        self.assertIn("embedding_model", stats)
        self.assertEqual(stats["documents"], 0)
        self.assertEqual(stats["chunks"], 0)


@unittest.skipIf(faiss is None, "FAISS not installed")
class TestCreateVectorStore(unittest.TestCase):
    def test_create_vector_store_returns_vector_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            from unittest.mock import patch
            with patch("offline_ai_assistant.data.vectorstore.Config") as cfg:
                cfg.FAISS_INDEX_PATH = Path(tmp) / "idx"
                cfg.SQLITE_DB_PATH = Path(tmp) / "db.sqlite"
                vs = create_vector_store(embedding_dim=8)
                self.assertIsInstance(vs, VectorStore)
                self.assertEqual(vs.embedding_dim, 8)
                vs.close()


if __name__ == "__main__":
    unittest.main()
