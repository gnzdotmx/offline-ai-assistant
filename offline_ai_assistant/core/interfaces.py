"""
Protocol interfaces for RAG pipeline components.

Using typing.Protocol allows dependency injection and easier testing
without requiring abstract base classes. Implementations live in
data/ (embedder, vectorstore, extractor, chunker) and llm/ (local_llm).
"""

from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Protocol, runtime_checkable

from .models import TextChunk, GenerationConfig


@runtime_checkable
class IChunker(Protocol):
    """Protocol for text chunking."""

    def chunk_text(
        self,
        text: str,
        source_file: str,
        preserve_structure: bool = True,
    ) -> List[TextChunk]:
        """Split text into chunks. Must return list of TextChunk."""
        ...


@runtime_checkable
class IEmbedder(Protocol):
    """Protocol for text embedding."""

    @property
    def embedding_dim(self) -> Optional[int]:
        """Dimension of embedding vectors."""
        ...

    def embed_text(self, text: str):
        """Embed a single text. Returns vector (e.g. numpy array)."""
        ...

    def embed_texts(self, texts: List[str], **kwargs):
        """Embed multiple texts. Returns array of vectors."""
        ...

    def embed_query(self, query: str):
        """Embed a query string. Returns vector."""
        ...

    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """Embed chunks and return list of dicts with 'embedding' and chunk metadata."""
        ...


@runtime_checkable
class IVectorStore(Protocol):
    """Protocol for vector storage and search."""

    def add_document(
        self,
        document_data: Dict[str, Any],
        chunks_data: List[Dict[str, Any]],
    ) -> int:
        """Add document and its chunk embeddings. Returns document_id."""
        ...

    def search(
        self,
        query_embedding: Any,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks. Returns list of hit dicts (text, score, file_name, etc.)."""
        ...

    def list_documents(self) -> List[Dict[str, Any]]:
        """List stored documents with metadata."""
        ...

    def delete_document(self, document_id: int) -> None:
        """Delete document and its chunks."""
        ...


@runtime_checkable
class IExtractor(Protocol):
    """Protocol for document text extraction."""

    def extract_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract text and metadata from file. Raises on error."""
        ...

    def validate_file(self, file_path: Path) -> tuple:
        """Validate file. Returns (is_valid: bool, error_message: str)."""
        ...


@runtime_checkable
class ILLM(Protocol):
    """Protocol for local LLM inference."""

    @property
    def model_path(self) -> Optional[Path]:
        """Path to loaded model."""
        ...

    def is_loaded(self) -> bool:
        """Return True if model is loaded."""
        ...

    def generate_complete(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate full response (non-streaming)."""
        ...

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Generate response token by token."""
        ...

    def create_rag_prompt(
        self,
        query: str,
        context_chunks: List[str],
        template: str,
    ) -> str:
        """Build RAG prompt from query and context."""
        ...

    def truncate_to_context(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit context window."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata dict."""
        ...


@runtime_checkable
class IReranker(Protocol):
    """Protocol for re-ranking retrieval candidates by relevance to the query."""

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Re-rank chunks by relevance to query. Returns up to top_k chunks."""
        ...
