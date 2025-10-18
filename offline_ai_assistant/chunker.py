"""
Text chunking module with token-aware splitting and overlap.

This module provides functionality to split text into chunks based on token counts
using tiktoken, with configurable overlap for better context preservation.
"""

import logging
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

try:
    import tiktoken
except ImportError:
    tiktoken = None

from .config import Config

logger = logging.getLogger("OfflineAIAssistant.chunker")


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    
    text: str # The actual chunk content
    start_char: int # Character position in original document
    end_char: int # End character position
    token_count: int # Number of tokens in this chunk
    chunk_index: int # Sequential chunk number
    source_file: str # Original file path
    source_section: Optional[str] = None # Section name (if applicable)
    overlap_with_previous: bool = False # Has overlap with previous chunk
    overlap_with_next: bool = False # Has overlap with next chunk


class TextChunker:
    """Token-aware text chunker with overlap support."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_model: str = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_model: Tiktoken encoding model to use
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.encoding_model = encoding_model or Config.ENCODING_MODEL
        
        if tiktoken is None:
            logger.error("tiktoken not installed. Falling back to simple word-based chunking.")
            self.tokenizer = None
        else:
            try:
                self.tokenizer = tiktoken.get_encoding(self.encoding_model)
            except Exception as e:
                logger.error(f"Error loading tiktoken encoding {self.encoding_model}: {e}")
                self.tokenizer = None
        
        logger.info(f"TextChunker initialized: chunk_size={self.chunk_size}, "
                   f"overlap={self.chunk_overlap}, encoding={self.encoding_model}")
    
    def chunk_text(
        self,
        text: str,
        source_file: str,
        preserve_structure: bool = True
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks based on token count.
        
        Args:
            text: Text to chunk
            source_file: Source file path for metadata
            preserve_structure: Whether to try preserving paragraph boundaries
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        logger.info(f"Chunking text from {source_file}: {len(text)} characters")
        
        if preserve_structure:
            return self._chunk_with_structure_preservation(text, source_file)
        else:
            return self._chunk_simple(text, source_file)
    
    def _chunk_with_structure_preservation(
        self,
        text: str,
        source_file: str
    ) -> List[TextChunk]:
        """
        Chunk text while trying to preserve paragraph and sentence boundaries.
        
        Args:
            text: Text to chunk
            source_file: Source file path for metadata
            
        Returns:
            List of TextChunk objects
        """
        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para_start, paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = self._count_tokens(test_chunk)
            
            if token_count <= self.chunk_size or not current_chunk:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = para_start
            else:
                # Finalize current chunk and start a new one
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk,
                        current_start,
                        current_start + len(current_chunk),
                        chunk_index,
                        source_file
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    # Adjust start position for overlap
                    current_start = para_start - len(overlap_text) - 2
                else:
                    current_chunk = paragraph
                    current_start = para_start
                
                # If single paragraph is too long, split it further
                if self._count_tokens(current_chunk) > self.chunk_size:
                    para_chunks = self._chunk_long_paragraph(
                        current_chunk,
                        current_start,
                        chunk_index,
                        source_file
                    )
                    chunks.extend(para_chunks)
                    chunk_index += len(para_chunks)
                    current_chunk = ""
        
        # Add final chunk if any text remains
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                chunk_index,
                source_file
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def _chunk_simple(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Simple chunking without structure preservation.
        
        Args:
            text: Text to chunk
            source_file: Source file path for metadata
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        words = text.split()
        current_chunk_words = []
        chunk_index = 0
        start_pos = 0
        
        for word in words:
            current_chunk_words.append(word)
            current_text = " ".join(current_chunk_words)
            
            if self._count_tokens(current_text) > self.chunk_size:
                # Remove the last word and create chunk
                current_chunk_words.pop()
                if current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
                    chunk = self._create_chunk(
                        chunk_text,
                        start_pos,
                        start_pos + len(chunk_text),
                        chunk_index,
                        source_file
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Handle overlap
                    if self.chunk_overlap > 0:
                        overlap_words = self._get_overlap_words(
                            current_chunk_words,
                            self.chunk_overlap
                        )
                        current_chunk_words = overlap_words + [word]
                    else:
                        current_chunk_words = [word]
                        start_pos += len(chunk_text) + 1
                else:
                    # Single word is too long, include it anyway
                    current_chunk_words = [word]
        
        # Add final chunk
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk = self._create_chunk(
                chunk_text,
                start_pos,
                start_pos + len(chunk_text),
                chunk_index,
                source_file
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[tuple]:
        """
        Split text into paragraphs with their start positions.
        
        Args:
            text: Text to split
            
        Returns:
            List of (start_position, paragraph_text) tuples
        """
        paragraphs = []
        current_pos = 0
        
        # Split on double newlines or more
        para_pattern = r'\n\s*\n'
        parts = re.split(para_pattern, text)
        
        for part in parts:
            part = part.strip()
            if part:
                # Find the actual position in original text
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
        source_file: str
    ) -> List[TextChunk]:
        """
        Split a long paragraph into smaller chunks.
        
        Args:
            paragraph: Paragraph text to split
            start_pos: Starting position in original text
            start_index: Starting chunk index
            source_file: Source file path
            
        Returns:
            List of TextChunk objects
        """
        # Try to split on sentences first
        sentences = self._split_into_sentences(paragraph)
        chunks = []
        current_chunk = ""
        current_start = start_pos
        chunk_index = start_index
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._count_tokens(test_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk,
                        current_start,
                        current_start + len(current_chunk),
                        chunk_index,
                        source_file
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                    current_start += len(current_chunk) + 1
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                chunk_index,
                source_file
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on common sentence endings
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(
        self,
        text: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
        source_file: str
    ) -> TextChunk:
        """
        Create a TextChunk object.
        
        Args:
            text: Chunk text
            start_char: Starting character position
            end_char: Ending character position
            chunk_index: Index of the chunk
            source_file: Source file path
            
        Returns:
            TextChunk object
        """
        token_count = self._count_tokens(text)
        
        return TextChunk(
            text=text.strip(),
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            chunk_index=chunk_index,
            source_file=source_file
        )
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Error counting tokens with tiktoken: {e}")
        
        # Fallback to word count approximation
        return len(text.split())
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens from text for overlap.
        
        Args:
            text: Source text
            overlap_tokens: Number of tokens to get
            
        Returns:
            Overlap text
        """
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text)
                if len(tokens) > overlap_tokens:
                    overlap_tokens_list = tokens[-overlap_tokens:]
                    return self.tokenizer.decode(overlap_tokens_list)
            except Exception as e:
                logger.warning(f"Error getting overlap with tiktoken: {e}")
        
        # Fallback to word-based overlap
        words = text.split()
        if len(words) > overlap_tokens:
            return " ".join(words[-overlap_tokens:])
        return text
    
    def _get_overlap_words(self, words: List[str], overlap_count: int) -> List[str]:
        """
        Get the last N words for overlap.
        
        Args:
            words: List of words
            overlap_count: Number of words to get
            
        Returns:
            List of overlap words
        """
        if len(words) > overlap_count:
            return words[-overlap_count:]
        return words.copy()
    
    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunk statistics
        """
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
            "max_chars": max(char_counts)
        }


def chunk_document(
    text: str,
    source_file: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[TextChunk]:
    """
    Convenience function to chunk document text.
    
    Args:
        text: Text to chunk
        source_file: Source file path
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of tokens to overlap
        
    Returns:
        List of TextChunk objects
    """
    chunker = TextChunker(chunk_size, chunk_overlap)
    return chunker.chunk_text(text, source_file)
