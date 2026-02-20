"""
Text chunking module with token-aware splitting and overlap.

Uses config for defaults and core.models for TextChunk.
"""

import logging
import re
from typing import List, Dict, Any, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None

from ..config import Config
from ..core.models import TextChunk

logger = logging.getLogger("OfflineAIAssistant.chunker")


class TextChunker:
    """Token-aware text chunker with overlap support."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_model: str = None,
    ):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.encoding_model = encoding_model or Config.ENCODING_MODEL

        if tiktoken is None:
            logger.warning(
                "tiktoken not installed. Using word-based chunking; chunk sizes are approximate. "
                "Install tiktoken for accurate token counts and better context-window behavior."
            )
            self.tokenizer = None
        else:
            try:
                self.tokenizer = tiktoken.get_encoding(self.encoding_model)
            except Exception as e:
                logger.warning(
                    "Failed to load tiktoken encoding %s: %s. Using word-based chunking; "
                    "chunk sizes are approximate.",
                    self.encoding_model,
                    e,
                )
                self.tokenizer = None

        # When using word-based fallback, use a conservative effective size so chunks stay under context
        ratio = getattr(Config, "WORD_FALLBACK_CHUNK_RATIO", 0.5)
        if self.tokenizer is None:
            self._effective_chunk_size = max(64, int(self.chunk_size * ratio))
            self._effective_chunk_overlap = min(
                int(self.chunk_overlap * ratio), max(0, self._effective_chunk_size - 1)
            )
        else:
            self._effective_chunk_size = self.chunk_size
            self._effective_chunk_overlap = self.chunk_overlap

        logger.info(
            "TextChunker initialized: chunk_size=%s, overlap=%s, encoding=%s",
            self._effective_chunk_size,
            self._effective_chunk_overlap,
            self.encoding_model,
        )

    def chunk_text(
        self,
        text: str,
        source_file: str,
        preserve_structure: bool = True,
    ) -> List[TextChunk]:
        if not text.strip():
            return []

        logger.info("Chunking text from %s: %s characters", source_file, len(text))

        if preserve_structure:
            return self._chunk_with_structure_preservation(text, source_file)
        return self._chunk_simple(text, source_file)

    def _chunk_with_structure_preservation(self, text: str, source_file: str) -> List[TextChunk]:
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for para_start, paragraph in paragraphs:
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = self._count_tokens(test_chunk)

            if token_count <= self._effective_chunk_size or not current_chunk:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = para_start
            else:
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk,
                        current_start,
                        current_start + len(current_chunk),
                        chunk_index,
                        source_file,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                if self._effective_chunk_overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(
                        current_chunk, self._effective_chunk_overlap
                    )
                    current_chunk = overlap_text + "\n\n" + paragraph
                    current_start = para_start - len(overlap_text) - 2
                else:
                    current_chunk = paragraph
                    current_start = para_start

                if self._count_tokens(current_chunk) > self._effective_chunk_size:
                    para_chunks = self._chunk_long_paragraph(
                        current_chunk, current_start, chunk_index, source_file
                    )
                    chunks.extend(para_chunks)
                    chunk_index += len(para_chunks)
                    current_chunk = ""

        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                chunk_index,
                source_file,
            )
            chunks.append(chunk)

        logger.info("Created %s chunks from %s", len(chunks), source_file)
        return chunks

    def _chunk_simple(self, text: str, source_file: str) -> List[TextChunk]:
        chunks = []
        words = text.split()
        current_chunk_words = []
        chunk_index = 0
        start_pos = 0

        for word in words:
            current_chunk_words.append(word)
            current_text = " ".join(current_chunk_words)

            if self._count_tokens(current_text) > self._effective_chunk_size:
                current_chunk_words.pop()
                if current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
                    chunk = self._create_chunk(
                        chunk_text,
                        start_pos,
                        start_pos + len(chunk_text),
                        chunk_index,
                        source_file,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    if self._effective_chunk_overlap > 0:
                        overlap_words = self._get_overlap_words(
                            current_chunk_words, self._effective_chunk_overlap
                        )
                        current_chunk_words = overlap_words + [word]
                    else:
                        current_chunk_words = [word]
                        start_pos += len(chunk_text) + 1
                else:
                    current_chunk_words = [word]

        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk = self._create_chunk(
                chunk_text,
                start_pos,
                start_pos + len(chunk_text),
                chunk_index,
                source_file,
            )
            chunks.append(chunk)

        logger.info("Created %s chunks from %s", len(chunks), source_file)
        return chunks

    def _split_into_paragraphs(self, text: str) -> List[tuple]:
        paragraphs = []
        current_pos = 0
        para_pattern = r"\n\s*\n"
        parts = re.split(para_pattern, text)

        for part in parts:
            part = part.strip()
            if part:
                start_pos = text.find(part, current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                paragraphs.append((start_pos, part))
                current_pos = start_pos + len(part)

        return paragraphs

    def _chunk_long_paragraph(
        self,
        paragraph: str,
        start_pos: int,
        start_index: int,
        source_file: str,
    ) -> List[TextChunk]:
        sentences = self._split_into_sentences(paragraph)
        chunks = []
        current_chunk = ""
        current_start = start_pos
        chunk_index = start_index

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if self._count_tokens(test_chunk) <= self._effective_chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk,
                        current_start,
                        current_start + len(current_chunk),
                        chunk_index,
                        source_file,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                if self._effective_chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(
                        current_chunk, self._effective_chunk_overlap
                    )
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                    current_start += len(current_chunk) + 1

        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                chunk_index,
                source_file,
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_pattern = r"[.!?]+\s+"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk(
        self,
        text: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
        source_file: str,
    ) -> TextChunk:
        token_count = self._count_tokens(text)
        return TextChunk(
            text=text.strip(),
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            chunk_index=chunk_index,
            source_file=source_file,
        )

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning("Error counting tokens with tiktoken: %s", e)
        return len(text.split())

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text)
                if len(tokens) > overlap_tokens:
                    return self.tokenizer.decode(tokens[-overlap_tokens:])
            except Exception as e:
                logger.warning("Error getting overlap with tiktoken: %s", e)
        words = text.split()
        if len(words) > overlap_tokens:
            return " ".join(words[-overlap_tokens:])
        return text

    def _get_overlap_words(self, words: List[str], overlap_count: int) -> List[str]:
        if len(words) > overlap_count:
            return words[-overlap_count:]
        return words.copy()

    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        if not chunks:
            return {}
        token_counts = [chunk.token_count for chunk in chunks]
        char_counts = [len(chunk.text) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "total_characters": sum(char_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
            "avg_chars_per_chunk": sum(char_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
        }


def chunk_document(
    text: str,
    source_file: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[TextChunk]:
    """Convenience function to chunk document text."""
    chunker = TextChunker(chunk_size, chunk_overlap)
    return chunker.chunk_text(text, source_file)
