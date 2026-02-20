"""
Data layer: document extraction, chunking, embeddings, vector store, model management.

All I/O and storage components live here. They depend on config and core models only.
"""

from .chunker import TextChunker, chunk_document
from .extractor import DocumentExtractor, extract_document
from .embedder import TextEmbedder, EmbeddingCache, create_embedder
from .vectorstore import VectorStore, create_vector_store
from .model_manager import ModelManager, ModelInfo, create_model_manager

__all__ = [
    "TextChunker",
    "chunk_document",
    "DocumentExtractor",
    "extract_document",
    "TextEmbedder",
    "EmbeddingCache",
    "create_embedder",
    "VectorStore",
    "create_vector_store",
    "ModelManager",
    "ModelInfo",
    "create_model_manager",
]
