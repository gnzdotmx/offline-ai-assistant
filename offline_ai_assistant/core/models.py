"""
Shared domain models used across the RAG pipeline, LLM, and UI.

Centralizing these dataclasses here keeps a single source of truth
and avoids circular imports between data, llm, and core layers.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    text: str
    start_char: int
    end_char: int
    token_count: int
    chunk_index: int
    source_file: str
    source_section: Optional[str] = None
    overlap_with_previous: bool = False
    overlap_with_next: bool = False


@dataclass
class RAGResult:
    """Result from RAG query with metadata."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    generation_time: float
    retrieval_time: float
    total_time: float
    tokens_generated: int
    chunks_retrieved: int
    model_used: str
    template_used: str


@dataclass
class ProcessingResult:
    """Result from document processing."""

    success: bool
    document_id: Optional[int]
    file_path: str
    chunks_created: int
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for LLM text generation."""

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = None
    stream: bool = True

    def __post_init__(self) -> None:
        if self.stop_sequences is None:
            self.stop_sequences = []
