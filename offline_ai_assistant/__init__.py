"""
Offline AI Assistant - A fully offline desktop AI assistant for document analysis.

This package provides a complete RAG (Retrieval-Augmented Generation) system
that runs entirely offline, ensuring complete privacy and security.

Structure:
  config/   - Configuration with validation and secure path handling
  core/      - Domain models, interfaces, and RAG pipeline
  data/      - Document extraction, chunking, embeddings, vector store, model manager
  llm/       - Local LLM (llama-cpp-python)
"""

__version__ = "1.0.0"
__author__ = "gnzdotmx"
__email__ = "gnzdotmxpj@gmail.com"
__description__ = "Fully offline desktop AI assistant for document analysis and question answering"

from .config import Config, setup_logging
from .core import (
    RAGPipeline,
    create_rag_pipeline,
    RAGResult,
    ProcessingResult,
    TextChunk,
    GenerationConfig,
)
from .core.interfaces import IEmbedder, IVectorStore, ILLM, IExtractor, IChunker
from .data import (
    TextChunker,
    DocumentExtractor,
    TextEmbedder,
    VectorStore,
    ModelManager,
    ModelInfo,
)
from .llm import LocalLLM, LLMManager, create_llm

__all__ = [
    "Config",
    "setup_logging",
    "RAGPipeline",
    "create_rag_pipeline",
    "RAGResult",
    "ProcessingResult",
    "TextChunk",
    "GenerationConfig",
    "IEmbedder",
    "IVectorStore",
    "ILLM",
    "IExtractor",
    "IChunker",
    "TextChunker",
    "DocumentExtractor",
    "TextEmbedder",
    "VectorStore",
    "ModelManager",
    "ModelInfo",
    "LocalLLM",
    "LLMManager",
    "create_llm",
]
