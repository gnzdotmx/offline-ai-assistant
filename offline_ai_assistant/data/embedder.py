"""
Text embedding module using sentence-transformers.

Uses config for model and cache paths. Cache files are under app-controlled dirs only (safe for pickle).
"""

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..config import Config
from ..core.models import TextChunk

logger = logging.getLogger("OfflineAIAssistant.embedder")


class TextEmbedder:
    """Generate embeddings for text using sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
        embedding_cache: Optional["EmbeddingCache"] = None,
    ):
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        self.cache_dir = cache_dir or Config.EMBEDDING_MODEL_CACHE
        self.device = device
        self.model = None
        self.embedding_dim = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if embedding_cache is not None:
            self.embedding_cache = embedding_cache
        elif Config.DISABLE_EMBEDDING_CACHE:
            self.embedding_cache = None
        else:
            cache_subdir = self.cache_dir / "embedding_cache"
            cache_subdir.mkdir(parents=True, exist_ok=True)
            self.embedding_cache = EmbeddingCache(cache_subdir)
        self._load_model()
        logger.info(
            "TextEmbedder initialized: model=%s, device=%s, dim=%s, cache=%s",
            self.model_name,
            self.device,
            self.embedding_dim,
            "disabled" if self.embedding_cache is None else "enabled",
        )

    def _load_model(self) -> None:
        if SentenceTransformer is None:
            logger.error("sentence-transformers not installed. Embeddings will not work.")
            return
        try:
            logger.info("Loading embedding model: %s", self.model_name)
            start_time = time.time()
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device,
            )
            test_embedding = self.model.encode("test", show_progress_bar=False)
            self.embedding_dim = len(test_embedding)
            logger.info(
                "Model loaded successfully in %.2fs. Embedding dimension: %s",
                time.time() - start_time,
                self.embedding_dim,
            )
        except Exception as e:
            logger.error("Error loading embedding model %s: %s", self.model_name, e)
            self.model = None
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

    def embed_text(self, text: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        if not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        if self.embedding_cache is not None:
            text_hash = self.get_embedding_hash(text)
            cached = self.embedding_cache.get(text_hash)
            if cached is not None:
                return np.array(cached, dtype=np.float32, copy=True)
        try:
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embedding = embedding.astype(np.float32)
            if self.embedding_cache is not None:
                self.embedding_cache.put(text_hash, embedding)
            return embedding
        except Exception as e:
            logger.error("Error generating embedding: %s", e)
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        logger.info("Generating embeddings for %s texts", len(texts))
        start_time = time.time()
        full_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]

        if not valid_indices:
            return full_embeddings

        if self.embedding_cache is not None:
            miss_indices = []
            miss_texts = []
            miss_hashes = []
            for i in valid_indices:
                text = texts[i]
                text_hash = self.get_embedding_hash(text)
                cached = self.embedding_cache.get(text_hash)
                if cached is not None:
                    full_embeddings[i] = cached
                else:
                    miss_indices.append(i)
                    miss_texts.append(text)
                    miss_hashes.append(text_hash)
            if not miss_texts:
                logger.info(
                    "Served %s embeddings from cache in %.2fs",
                    len(texts),
                    time.time() - start_time,
                )
                return full_embeddings
            valid_texts = miss_texts
            valid_indices = miss_indices
            encode_hashes = miss_hashes
        else:
            valid_texts = [texts[i] for i in valid_indices]
            encode_hashes = None

        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)
        for j, valid_idx in enumerate(valid_indices):
            full_embeddings[valid_idx] = embeddings[j]
            if self.embedding_cache is not None and encode_hashes is not None:
                self.embedding_cache.put(encode_hashes[j], embeddings[j])
        logger.info(
            "Generated %s embeddings in %.2fs (%.1f texts/sec)",
            len(texts),
            time.time() - start_time,
            len(texts) / (time.time() - start_time or 1e-6),
        )
        return full_embeddings

    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        logger.info("Embedding %s text chunks", len(chunks))
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size, show_progress)
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunks.append({
                "text": chunk.text,
                "embedding": embeddings[i],
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
                "source_file": chunk.source_file,
                "source_section": chunk.source_section,
                "embedding_model": self.model_name,
                "embedding_dim": self.embedding_dim,
            })
        return embedded_chunks

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string; uses same cache as embed_text for repeated/similar queries."""
        logger.debug("Embedding query: %s...", query[:100])
        return self.embed_text(query)

    def calculate_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        if len(document_embeddings) == 0:
            return np.array([])
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )
        return np.dot(doc_norms, query_norm)

    def save_embeddings(self, embeddings: List[Dict[str, Any]], file_path: Path) -> None:
        try:
            with open(file_path, "wb") as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved %s embeddings to %s", len(embeddings), file_path)
        except Exception as e:
            logger.error("Error saving embeddings to %s: %s", file_path, e)
            raise RuntimeError(f"Failed to save embeddings: {e}") from e

    def load_embeddings(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load embeddings from disk. Only use with app-controlled paths (safe for pickle)."""
        try:
            with open(file_path, "rb") as f:
                embeddings = pickle.load(f)
            logger.info("Loaded %s embeddings from %s", len(embeddings), file_path)
            return embeddings
        except Exception as e:
            logger.error("Error loading embeddings from %s: %s", file_path, e)
            raise RuntimeError(f"Failed to load embeddings: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "not_loaded"}
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_seq_length": getattr(self.model, "max_seq_length", None),
            "cache_dir": str(self.cache_dir),
            "embedding_cache_enabled": self.embedding_cache is not None,
        }

    def validate_embeddings(self, embeddings: List[Dict[str, Any]]) -> tuple:
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
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()


class EmbeddingCache:
    """Cache for storing and retrieving embeddings (files under app-controlled dirs only)."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "embedding_cache.pkl"
        self.cache = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info("Loaded embedding cache with %s entries", len(self.cache))
            except Exception as e:
                logger.warning("Error loading embedding cache: %s", e)
                self.cache = {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error("Error saving embedding cache: %s", e)

    def get(self, text_hash: str) -> Optional[np.ndarray]:
        return self.cache.get(text_hash)

    def put(self, text_hash: str, embedding: np.ndarray) -> None:
        self.cache[text_hash] = embedding
        if len(self.cache) % 100 == 0:
            self._save_cache()

    def clear(self) -> None:
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding cache cleared")


def create_embedder(
    model_name: Optional[str] = None,
    device: str = "cpu",
) -> TextEmbedder:
    return TextEmbedder(model_name=model_name, device=device)
