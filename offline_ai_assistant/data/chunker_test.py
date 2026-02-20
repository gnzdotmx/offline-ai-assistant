"""
Tests for offline_ai_assistant.data.chunker.

Uses word-based chunking (tiktoken patched to None or failing) for predictable
token counts. Covers chunk_text, get_chunk_stats, and chunk_document.
"""

import unittest
from unittest.mock import patch, MagicMock

from offline_ai_assistant.data.chunker import TextChunker, chunk_document
from offline_ai_assistant.core.models import TextChunk


def _make_chunker_word_based(chunk_size: int = 10, chunk_overlap: int = 2):
    """Create a TextChunker that uses word-based token counting (no tiktoken)."""
    with patch("offline_ai_assistant.data.chunker.tiktoken", None):
        with patch("offline_ai_assistant.data.chunker.Config") as cfg:
            cfg.CHUNK_SIZE = chunk_size
            cfg.CHUNK_OVERLAP = chunk_overlap
            cfg.ENCODING_MODEL = "cl100k_base"
            cfg.WORD_FALLBACK_CHUNK_RATIO = 1.0
            return TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_model="cl100k_base",
            )


class TestTextChunkerInit(unittest.TestCase):
    def test_init_uses_config_defaults_when_tiktoken_missing(self):
        with patch("offline_ai_assistant.data.chunker.tiktoken", None):
            with patch("offline_ai_assistant.data.chunker.Config") as cfg:
                cfg.CHUNK_SIZE = 512
                cfg.CHUNK_OVERLAP = 50
                cfg.ENCODING_MODEL = "cl100k_base"
                cfg.WORD_FALLBACK_CHUNK_RATIO = 0.5
                c = TextChunker()
                self.assertIsNone(c.tokenizer)
                self.assertEqual(c.chunk_size, 512)
                self.assertEqual(c._effective_chunk_size, max(64, 512 * 0.5))

    def test_init_falls_back_to_word_based_when_tiktoken_get_encoding_fails(self):
        with patch("offline_ai_assistant.data.chunker.tiktoken") as mock_tiktoken:
            mock_tiktoken.get_encoding.side_effect = ValueError("Unknown encoding")
            with patch("offline_ai_assistant.data.chunker.Config") as cfg:
                cfg.CHUNK_SIZE = 512
                cfg.CHUNK_OVERLAP = 50
                cfg.ENCODING_MODEL = "cl100k_base"
                cfg.WORD_FALLBACK_CHUNK_RATIO = 0.5
                c = TextChunker()
                self.assertIsNone(c.tokenizer)
                self.assertEqual(c._effective_chunk_size, max(64, int(512 * 0.5)))


class TestChunkTextEmpty(unittest.TestCase):
    def test_empty_string_returns_empty_list(self):
        chunker = _make_chunker_word_based(chunk_size=10, chunk_overlap=2)
        self.assertEqual(chunker.chunk_text("", "file.txt"), [])
        self.assertEqual(chunker.chunk_text("   \n\n  ", "file.txt"), [])


class TestChunkTextSimple(unittest.TestCase):
    def test_short_text_single_chunk(self):
        chunker = _make_chunker_word_based(chunk_size=100, chunk_overlap=0)
        text = "One two three four five."
        chunks = chunker.chunk_text(text, "doc.txt", preserve_structure=False)
        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], TextChunk)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].source_file, "doc.txt")
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[0].end_char, len(text))
        self.assertEqual(chunks[0].token_count, 5)

    def test_multiple_chunks_when_over_size(self):
        # Word-based fallback uses min 64 tokens per chunk; use 120+ words to get 2 chunks
        chunker = _make_chunker_word_based(chunk_size=100, chunk_overlap=0)
        words = ["w" + str(i) for i in range(120)]
        text = " ".join(words)
        chunks = chunker.chunk_text(text, "doc.txt", preserve_structure=False)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        for i, c in enumerate(chunks):
            self.assertEqual(c.chunk_index, i)
            self.assertEqual(c.source_file, "doc.txt")

    def test_preserve_structure_True_uses_paragraphs(self):
        chunker = _make_chunker_word_based(chunk_size=100, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunker.chunk_text(text, "doc.txt", preserve_structure=True)
        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertIsInstance(c, TextChunk)
            self.assertEqual(c.source_file, "doc.txt")


class TestChunkTextFields(unittest.TestCase):
    def test_chunk_has_required_fields(self):
        chunker = _make_chunker_word_based(chunk_size=20, chunk_overlap=0)
        text = "Hello world. How are you?"
        chunks = chunker.chunk_text(text, "x.pdf", preserve_structure=False)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIsInstance(c.text, str)
            self.assertIsInstance(c.start_char, int)
            self.assertIsInstance(c.end_char, int)
            self.assertIsInstance(c.token_count, int)
            self.assertIsInstance(c.chunk_index, int)
            self.assertEqual(c.source_file, "x.pdf")
            self.assertGreaterEqual(c.token_count, 0)
            self.assertLessEqual(c.start_char, c.end_char)


class TestGetChunkStats(unittest.TestCase):
    def test_empty_chunks_returns_empty_dict(self):
        chunker = _make_chunker_word_based()
        self.assertEqual(chunker.get_chunk_stats([]), {})

    def test_returns_aggregate_stats(self):
        chunker = _make_chunker_word_based()
        text = "one two three four five six seven eight nine ten"
        chunks = chunker.chunk_text(text, "doc.txt", preserve_structure=False)
        stats = chunker.get_chunk_stats(chunks)
        self.assertEqual(stats["total_chunks"], len(chunks))
        self.assertIn("total_tokens", stats)
        self.assertIn("total_characters", stats)
        self.assertIn("avg_tokens_per_chunk", stats)
        self.assertIn("min_tokens", stats)
        self.assertIn("max_tokens", stats)
        self.assertIn("min_chars", stats)
        self.assertIn("max_chars", stats)
        self.assertGreater(stats["total_tokens"], 0)


class TestChunkDocument(unittest.TestCase):
    def test_chunk_document_returns_list_of_chunks(self):
        with patch("offline_ai_assistant.data.chunker.Config") as cfg:
            cfg.CHUNK_SIZE = 512
            cfg.CHUNK_OVERLAP = 50
            cfg.ENCODING_MODEL = "cl100k_base"
            cfg.WORD_FALLBACK_CHUNK_RATIO = 1.0
            with patch("offline_ai_assistant.data.chunker.tiktoken", None):
                result = chunk_document("Short text here.", "file.txt")
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 1)
                self.assertIsInstance(result[0], TextChunk)


class TestCountTokens(unittest.TestCase):
    def test_word_based_count_equals_word_count(self):
        chunker = _make_chunker_word_based()
        self.assertEqual(chunker._count_tokens("a b c"), 3)
        self.assertEqual(chunker._count_tokens("single"), 1)

    def test_count_tokens_falls_back_to_words_when_tokenizer_raises(self):
        chunker = _make_chunker_word_based()
        with patch.object(chunker, "tokenizer", MagicMock()) as mock_tok:
            mock_tok.encode.side_effect = RuntimeError("encode failed")
            self.assertEqual(chunker._count_tokens("one two three"), 3)


class TestSplitIntoParagraphs(unittest.TestCase):
    def test_splits_on_double_newline(self):
        chunker = _make_chunker_word_based()
        paras = chunker._split_into_paragraphs("A\n\nB\n\nC")
        self.assertEqual(len(paras), 3)
        self.assertEqual(paras[0][1], "A")
        self.assertEqual(paras[1][1], "B")
        self.assertEqual(paras[2][1], "C")

    def test_ignores_empty_parts(self):
        chunker = _make_chunker_word_based()
        paras = chunker._split_into_paragraphs("A\n\n\n\nB")
        self.assertEqual(len(paras), 2)


class TestSplitIntoSentences(unittest.TestCase):
    def test_splits_on_sentence_endings(self):
        chunker = _make_chunker_word_based()
        sentences = chunker._split_into_sentences("First. Second. Third.")
        self.assertGreaterEqual(len(sentences), 1)
        self.assertTrue(all(isinstance(s, str) for s in sentences))


class TestGetOverlapText(unittest.TestCase):
    def test_get_overlap_words_returns_tail_when_long_enough(self):
        chunker = _make_chunker_word_based(chunk_size=10, chunk_overlap=2)
        words = ["a", "b", "c", "d", "e"]
        out = chunker._get_overlap_words(words, 2)
        self.assertEqual(out, ["d", "e"])

    def test_get_overlap_words_returns_copy_when_short(self):
        chunker = _make_chunker_word_based()
        words = ["a", "b"]
        out = chunker._get_overlap_words(words, 5)
        self.assertEqual(out, ["a", "b"])
        self.assertIsNot(out, words)
