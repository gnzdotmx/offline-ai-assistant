"""
Tests for offline_ai_assistant.data.embedder.

Uses mocks for SentenceTransformer so no real model is loaded. Covers
TextEmbedder (embed_text, embed_texts, embed_chunks, embed_query,
calculate_similarity, validate_embeddings, save/load, get_embedding_hash),
EmbeddingCache, and create_embedder.
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from offline_ai_assistant.data.embedder import (
    TextEmbedder,
    EmbeddingCache,
    create_embedder,
)
from offline_ai_assistant.core.models import TextChunk


def _make_mock_embedder(embedding_dim: int = 384, use_cache: bool = False):
    """Create a TextEmbedder with a mocked SentenceTransformer model."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda x, **kw: (
        np.ones(embedding_dim, dtype=np.float32) / np.sqrt(embedding_dim)
        if isinstance(x, str)
        else np.ones((len(x), embedding_dim), dtype=np.float32) / np.sqrt(embedding_dim)
    )
    mock_model.max_seq_length = 512

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with patch("offline_ai_assistant.data.embedder.SentenceTransformer") as st:
            st.return_value = mock_model
            with patch("offline_ai_assistant.data.embedder.Config") as cfg:
                cfg.EMBEDDING_MODEL_NAME = "test-model"
                cfg.EMBEDDING_MODEL_CACHE = cache_dir
                cfg.DISABLE_EMBEDDING_CACHE = not use_cache
                embedder = TextEmbedder(
                    model_name="test-model",
                    cache_dir=cache_dir,
                    embedding_cache=EmbeddingCache(cache_dir / "ec") if use_cache else None,
                )
        embedder.model = mock_model
        embedder.embedding_dim = embedding_dim
        if use_cache:
            embedder.embedding_cache = EmbeddingCache(cache_dir / "ec")
        else:
            embedder.embedding_cache = None
        return embedder


class TestTextEmbedderEmbedText(unittest.TestCase):
    def test_embed_text_returns_array_of_expected_dim(self):
        embedder = _make_mock_embedder(embedding_dim=384)
        out = embedder.embed_text("hello world")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (384,))

    def test_embed_text_empty_string_returns_zeros(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        out = embedder.embed_text("   ")
        np.testing.assert_array_equal(out, np.zeros(8, dtype=np.float32))

    def test_embed_text_model_none_raises(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        embedder.model = None
        with self.assertRaises(RuntimeError) as ctx:
            embedder.embed_text("hello")
        self.assertIn("not loaded", str(ctx.exception))


class TestTextEmbedderEmbedTexts(unittest.TestCase):
    def test_embed_texts_empty_list_returns_empty_2d_array(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        out = embedder.embed_texts([])
        self.assertEqual(out.shape, (0, 8))

    def test_embed_texts_returns_2d_array(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        out = embedder.embed_texts(["a", "b", "c"])
        self.assertEqual(out.shape, (3, 8))
        self.assertEqual(out.dtype, np.float32)


class TestTextEmbedderEmbedChunks(unittest.TestCase):
    def test_embed_chunks_empty_returns_empty_list(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        self.assertEqual(embedder.embed_chunks([]), [])

    def test_embed_chunks_returns_list_of_dicts_with_required_keys(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        chunks = [
            TextChunk("first", 0, 5, 2, 0, "doc.txt"),
            TextChunk("second", 6, 12, 2, 1, "doc.txt"),
        ]
        result = embedder.embed_chunks(chunks)
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertIn("text", r)
            self.assertIn("embedding", r)
            self.assertIn("start_char", r)
            self.assertIn("end_char", r)
            self.assertIn("token_count", r)
            self.assertIn("chunk_index", r)
            self.assertIn("source_file", r)
            self.assertIn("embedding_model", r)
            self.assertIn("embedding_dim", r)
            self.assertEqual(r["embedding"].shape, (8,))

    def test_embed_chunks_preserves_chunk_metadata(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        chunk = TextChunk("hello", 10, 15, 3, 0, "x.pdf", source_section="intro")
        result = embedder.embed_chunks([chunk])
        self.assertEqual(result[0]["text"], "hello")
        self.assertEqual(result[0]["start_char"], 10)
        self.assertEqual(result[0]["end_char"], 15)
        self.assertEqual(result[0]["chunk_index"], 0)
        self.assertEqual(result[0]["source_file"], "x.pdf")
        self.assertEqual(result[0]["source_section"], "intro")


class TestTextEmbedderEmbedQuery(unittest.TestCase):
    def test_embed_query_delegates_to_embed_text(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        out = embedder.embed_query("query string")
        self.assertEqual(out.shape, (8,))


class TestTextEmbedderCalculateSimilarity(unittest.TestCase):
    def test_calculate_similarity_empty_docs_returns_empty_array(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        q = np.ones(8, dtype=np.float32)
        q = q / np.linalg.norm(q)
        out = embedder.calculate_similarity(q, np.array([]).reshape(0, 8))
        self.assertEqual(out.shape, (0,))

    def test_calculate_similarity_returns_cosine_scores(self):
        embedder = _make_mock_embedder(embedding_dim=4)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        docs = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
        out = embedder.calculate_similarity(q, docs)
        self.assertAlmostEqual(out[0], 1.0)
        self.assertAlmostEqual(out[1], 0.0)


class TestTextEmbedderGetEmbeddingHash(unittest.TestCase):
    def test_get_embedding_hash_deterministic(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        h1 = embedder.get_embedding_hash("same text")
        h2 = embedder.get_embedding_hash("same text")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h1))

    def test_get_embedding_hash_different_text_different_hash(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        self.assertNotEqual(
            embedder.get_embedding_hash("a"),
            embedder.get_embedding_hash("b"),
        )


class TestTextEmbedderValidateEmbeddings(unittest.TestCase):
    def test_validate_embeddings_empty_returns_false(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        ok, msg = embedder.validate_embeddings([])
        self.assertFalse(ok)
        self.assertIn("No embeddings", msg)

    def test_validate_embeddings_dim_mismatch_returns_false(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        bad = [{"embedding": np.ones(4), "embedding_dim": 4, "embedding_model": "test-model"}]
        ok, msg = embedder.validate_embeddings(bad)
        self.assertFalse(ok)
        self.assertIn("dimension", msg.lower())

    def test_validate_embeddings_model_mismatch_returns_false(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        bad = [{"embedding": np.ones(8), "embedding_dim": 8, "embedding_model": "other-model"}]
        ok, msg = embedder.validate_embeddings(bad)
        self.assertFalse(ok)
        self.assertIn("model", msg.lower())

    def test_validate_embeddings_valid_returns_true(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        good = [{"embedding": np.ones(8, dtype=np.float32), "embedding_dim": 8, "embedding_model": "test-model"}]
        ok, msg = embedder.validate_embeddings(good)
        self.assertTrue(ok)
        self.assertEqual(msg, "")


class TestTextEmbedderSaveLoadEmbeddings(unittest.TestCase):
    def test_save_and_load_embeddings_roundtrip(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        data = [
            {"embedding": np.ones(8), "embedding_dim": 8, "embedding_model": "test", "text": "a"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            embedder.save_embeddings(data, path)
            self.assertTrue(path.exists())
            loaded = embedder.load_embeddings(path)
            self.assertEqual(len(loaded), 1)
            np.testing.assert_array_almost_equal(loaded[0]["embedding"], data[0]["embedding"])
        finally:
            path.unlink(missing_ok=True)


class TestTextEmbedderGetModelInfo(unittest.TestCase):
    def test_get_model_info_when_loaded(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        info = embedder.get_model_info()
        self.assertEqual(info["status"], "loaded")
        self.assertEqual(info["model_name"], "test-model")
        self.assertEqual(info["embedding_dim"], 8)

    def test_get_model_info_when_not_loaded(self):
        embedder = _make_mock_embedder(embedding_dim=8)
        embedder.model = None
        info = embedder.get_model_info()
        self.assertEqual(info["status"], "not_loaded")


class TestEmbeddingCache(unittest.TestCase):
    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "ec")
            vec = np.ones(4, dtype=np.float32)
            cache.put("key1", vec)
            out = cache.get("key1")
            np.testing.assert_array_equal(out, vec)

    def test_get_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "ec")
            self.assertIsNone(cache.get("nonexistent"))

    def test_clear_empties_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "ec")
            cache.put("k", np.ones(4))
            cache.clear()
            self.assertIsNone(cache.get("k"))


class TestCreateEmbedder(unittest.TestCase):
    def test_create_embedder_returns_text_embedder(self):
        with patch("offline_ai_assistant.data.embedder.SentenceTransformer") as st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.ones(384, dtype=np.float32)
            st.return_value = mock_model
            with tempfile.TemporaryDirectory() as tmp:
                with patch("offline_ai_assistant.data.embedder.Config") as cfg:
                    cfg.EMBEDDING_MODEL_NAME = "test"
                    cfg.EMBEDDING_MODEL_CACHE = Path(tmp)
                    cfg.DISABLE_EMBEDDING_CACHE = True
                    embedder = create_embedder(model_name="test")
                    self.assertIsInstance(embedder, TextEmbedder)
