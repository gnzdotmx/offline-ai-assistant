"""
Text embedding module using sentence-transformers.

This module provides functionality to generate embeddings for text chunks
using local sentence-transformer models with caching and batch processing.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
import pickle
import hashlib
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .config import Config
from .chunker import TextChunk

logger = logging.getLogger("OfflineAIAssistant.embedder")


class TextEmbedder:
    """Generate embeddings for text using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = None,
        cache_dir: Path = None,
        device: str = "cpu"
    ):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
            cache_dir: Directory to cache the model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        self.cache_dir = cache_dir or Config.EMBEDDING_MODEL_CACHE
        self.device = device
        self.model = None
        self.embedding_dim = None
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._load_model()
        
        logger.info(f"TextEmbedder initialized: model={self.model_name}, "
                   f"device={self.device}, dim={self.embedding_dim}")
    
    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        if SentenceTransformer is None:
            logger.error("sentence-transformers not installed. Embeddings will not work.")
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device
            )
            
            # Get embedding dimension
            test_embedding = self.model.encode("test", show_progress_bar=False)
            self.embedding_dim = len(test_embedding)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s. "
                       f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model {self.model_name}: {e}")
            self.model = None
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        if not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()
        
        try:
            # Filter out empty texts but keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            
            # Generate embeddings for valid texts
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Create full embedding array with zeros for empty texts
            full_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            for i, valid_idx in enumerate(valid_indices):
                full_embeddings[valid_idx] = embeddings[i]
            
            embed_time = time.time() - start_time
            logger.info(f"Generated {len(texts)} embeddings in {embed_time:.2f}s "
                       f"({len(texts)/embed_time:.1f} texts/sec)")
            
            return full_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise RuntimeError(f"Failed to generate batch embeddings: {e}")
    
    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Number of chunks to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            List of dictionaries containing chunk data and embeddings
        """
        if not chunks:
            return []
        
        logger.info(f"Embedding {len(chunks)} text chunks")
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size, show_progress)
        
        # Combine chunks with embeddings
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = {
                "text": chunk.text,
                "embedding": embeddings[i],
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
                "source_file": chunk.source_file,
                "source_section": chunk.source_section,
                "embedding_model": self.model_name,
                "embedding_dim": self.embedding_dim
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Numpy array containing the query embedding
        """
        logger.debug(f"Embedding query: {query[:100]}...")
        return self.embed_text(query)
    
    def calculate_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Array of document embeddings
            
        Returns:
            Array of similarity scores
        """
        if len(document_embeddings) == 0:
            return np.array([])
        
        # Ensure embeddings are normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )
        
        # Calculate cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def save_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        file_path: Path
    ) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: List of embedding dictionaries
            file_path: Path to save the embeddings
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings to {file_path}: {e}")
            raise RuntimeError(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load embeddings from disk.
        
        Args:
            file_path: Path to load embeddings from
            
        Returns:
            List of embedding dictionaries
        """
        try:
            with open(file_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {e}")
            raise RuntimeError(f"Failed to load embeddings: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', None),
            "cache_dir": str(self.cache_dir)
        }
    
    def validate_embeddings(
        self,
        embeddings: List[Dict[str, Any]]
    ) -> tuple[bool, str]:
        """
        Validate embedding data.
        
        Args:
            embeddings: List of embedding dictionaries
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not embeddings:
            return False, "No embeddings provided"
        
        try:
            first_embedding = embeddings[0]
            expected_dim = first_embedding.get("embedding_dim")
            expected_model = first_embedding.get("embedding_model")
            
            if expected_dim != self.embedding_dim:
                return False, f"Embedding dimension mismatch: expected {self.embedding_dim}, got {expected_dim}"
            
            if expected_model != self.model_name:
                return False, f"Model mismatch: expected {self.model_name}, got {expected_model}"
            
            # Check all embeddings have correct shape
            for i, emb_data in enumerate(embeddings):
                embedding = emb_data.get("embedding")
                if embedding is None:
                    return False, f"Missing embedding at index {i}"
                
                if not isinstance(embedding, np.ndarray):
                    return False, f"Invalid embedding type at index {i}"
                
                if embedding.shape != (self.embedding_dim,):
                    return False, f"Invalid embedding shape at index {i}: {embedding.shape}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def get_embedding_hash(self, text: str) -> str:
        """
        Generate hash for text to cache embeddings.
        
        Args:
            text: Text to hash
            
        Returns:
            SHA-256 hash string
        """
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()


class EmbeddingCache:
    """Cache for storing and retrieving embeddings."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "embedding_cache.pkl"
        self.cache = {}
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded embedding cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Error loading embedding cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
    
    def get(self, text_hash: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text_hash: Hash of the text
            
        Returns:
            Cached embedding or None
        """
        return self.cache.get(text_hash)
    
    def put(self, text_hash: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            text_hash: Hash of the text
            embedding: Embedding to cache
        """
        self.cache[text_hash] = embedding
        
        # Periodically save cache
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding cache cleared")


def create_embedder(
    model_name: str = None,
    device: str = "cpu"
) -> TextEmbedder:
    """
    Convenience function to create a text embedder.
    
    Args:
        model_name: Name of the sentence-transformer model
        device: Device to run the model on
        
    Returns:
        TextEmbedder instance
    """
    return TextEmbedder(model_name=model_name, device=device)
