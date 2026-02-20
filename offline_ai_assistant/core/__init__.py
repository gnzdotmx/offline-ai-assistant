"""
Core domain models and RAG orchestration.

This package contains shared data models, protocol interfaces for
pluggable components, and the RAG pipeline.
"""

from .models import (
    RAGResult,
    ProcessingResult,
    TextChunk,
    GenerationConfig,
)
from .interfaces import (
    IEmbedder,
    IVectorStore,
    ILLM,
    IExtractor,
    IChunker,
)
from .rag import RAGPipeline, create_rag_pipeline

__all__ = [
    "RAGResult",
    "ProcessingResult",
    "TextChunk",
    "GenerationConfig",
    "IEmbedder",
    "IVectorStore",
    "ILLM",
    "IExtractor",
    "IChunker",
    "RAGPipeline",
    "create_rag_pipeline",
]
