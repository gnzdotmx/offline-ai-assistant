"""
Offline AI Assistant - A fully offline desktop AI assistant for document analysis.

This package provides a complete RAG (Retrieval-Augmented Generation) system
that runs entirely offline, ensuring complete privacy and security.
"""

__version__ = "1.0.0"
__author__ = "gnzdotmx"
__email__ = "gnzdotmxpj@gmail.com"
__description__ = "Fully offline desktop AI assistant for document analysis and question answering"

from .config import Config
from .rag import RAGPipeline, create_rag_pipeline
from .llm import LocalLLM, GenerationConfig
from .embedder import TextEmbedder
from .vectorstore import VectorStore
from .extractor import DocumentExtractor
from .chunker import TextChunker

__all__ = [
    "Config",
    "RAGPipeline", 
    "create_rag_pipeline",
    "LocalLLM",
    "GenerationConfig", 
    "TextEmbedder",
    "VectorStore",
    "DocumentExtractor",
    "TextChunker"
]
