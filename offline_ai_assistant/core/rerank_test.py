"""
Tests for offline_ai_assistant.core.rerank.

Covers _tokenize, _keyword_overlap_score, rerank, and no_op_rerank.
"""

import unittest
from unittest.mock import patch

from offline_ai_assistant.core.rerank import (
    _tokenize,
    _keyword_overlap_score,
    rerank,
    no_op_rerank,
)


class TestTokenize(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(_tokenize(""), [])
        self.assertEqual(_tokenize(None), [])

    def test_lowercase(self):
        self.assertEqual(_tokenize("Hello World"), ["hello", "world"])

    def test_alphanumeric_only(self):
        self.assertEqual(_tokenize("abc 123 x1y2"), ["abc", "123", "x1y2"])

    def test_splits_on_non_alphanumeric(self):
        self.assertEqual(_tokenize("one-two.three"), ["one", "two", "three"])

    def test_ignores_punctuation_only(self):
        self.assertEqual(_tokenize("... --- ..."), [])


class TestKeywordOverlapScore(unittest.TestCase):
    def test_empty_query_tokens_returns_zero(self):
        self.assertEqual(_keyword_overlap_score([], "some chunk text"), 0.0)

    def test_empty_chunk_returns_zero(self):
        self.assertEqual(_keyword_overlap_score(["query", "terms"], ""), 0.0)

    def test_no_overlap_returns_zero(self):
        self.assertEqual(
            _keyword_overlap_score(["apple", "banana"], "chunk about dogs"),
            0.0,
        )

    def test_full_overlap_normalized_by_query_length(self):
        score = _keyword_overlap_score(["a", "b"], "a b")
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
        self.assertAlmostEqual(score, 2.0 / 2)  # both match, / len(query)

    def test_partial_overlap(self):
        score = _keyword_overlap_score(["hello", "world"], "hello there")
        self.assertGreater(score, 0)
        self.assertLess(score, 1.0)

    def test_repeated_query_term_in_chunk_boosts_score(self):
        # "cat" appears multiple times in chunk
        score_multi = _keyword_overlap_score(["cat"], "cat cat cat")
        score_single = _keyword_overlap_score(["cat"], "cat")
        self.assertGreater(score_multi, score_single)


class TestRerank(unittest.TestCase):
    def test_empty_chunks_returns_empty(self):
        self.assertEqual(rerank("query", [], 5), [])

    def test_top_k_zero_returns_empty(self):
        chunks = [{"text": "a", "score": 0.9}]
        self.assertEqual(rerank("query", chunks, 0), [])

    def test_when_rag_rerank_false_returns_slice_unchanged(self):
        chunks = [
            {"text": "first", "score": 0.5},
            {"text": "second", "score": 0.9},
            {"text": "third", "score": 0.7},
        ]
        with patch("offline_ai_assistant.core.rerank.Config") as cfg:
            cfg.RAG_RERANK = False
            result = rerank("any query", chunks, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "first")
        self.assertEqual(result[1]["text"], "second")
        self.assertNotIn("rerank_score", result[0])

    def test_when_rag_rerank_true_reorders_by_keyword_score(self):
        chunks = [
            {"text": "unrelated content here"},
            {"text": "python programming language"},
            {"text": "python and coding"},
        ]
        with patch("offline_ai_assistant.core.rerank.Config") as cfg:
            cfg.RAG_RERANK = True
            result = rerank("python", chunks, 3)
        self.assertEqual(len(result), 3)
        self.assertIn("rerank_score", result[0])
        # Chunks mentioning "python" should rank higher
        scores = [r["rerank_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rerank_adds_rerank_score_to_each(self):
        chunks = [{"text": "hello world", "score": 0.8}]
        with patch("offline_ai_assistant.core.rerank.Config") as cfg:
            cfg.RAG_RERANK = True
            result = rerank("hello", chunks, 1)
        self.assertEqual(len(result), 1)
        self.assertIn("rerank_score", result[0])
        self.assertGreater(result[0]["rerank_score"], 0)

    def test_empty_query_tokens_returns_slice_unchanged(self):
        chunks = [{"text": "a"}, {"text": "b"}]
        with patch("offline_ai_assistant.core.rerank.Config") as cfg:
            cfg.RAG_RERANK = True
            result = rerank("...!!!...", chunks, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "a")


class TestNoOpRerank(unittest.TestCase):
    def test_returns_slice_unchanged(self):
        chunks = [
            {"text": "a", "score": 0.1},
            {"text": "b", "score": 0.9},
            {"text": "c", "score": 0.5},
        ]
        result = no_op_rerank("query", chunks, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "a")
        self.assertEqual(result[1]["text"], "b")

    def test_top_k_larger_than_chunks_returns_all(self):
        chunks = [{"text": "only"}]
        result = no_op_rerank("q", chunks, 10)
        self.assertEqual(len(result), 1)

    def test_empty_chunks_returns_empty(self):
        self.assertEqual(no_op_rerank("q", [], 5), [])


if __name__ == "__main__":
    unittest.main()
