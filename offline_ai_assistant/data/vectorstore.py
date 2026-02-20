"""
Vector store module using FAISS for vector search and SQLite for metadata.

Uses config for paths and defaults. No dependency on embedder; embedding_dim passed in or from DB.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from ..config import Config

logger = logging.getLogger("OfflineAIAssistant.vectorstore")

# Default embedding dimension when not provided (e.g. all-MiniLM-L6-v2)
DEFAULT_EMBEDDING_DIM = 384


class VectorStore:
    """Vector store combining FAISS index with SQLite metadata."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.index_path = index_path or Config.FAISS_INDEX_PATH
        self.db_path = db_path or Config.SQLITE_DB_PATH
        self.embedding_dim = embedding_dim
        self.index = None
        self.index_id_map = {}
        self.conn = None
        self._check_dependencies()
        self._initialize_database()
        self._load_or_create_index()
        logger.info(
            "VectorStore initialized: index=%s, db=%s",
            self.index_path,
            self.db_path,
        )

    def _check_dependencies(self) -> None:
        if faiss is None:
            raise RuntimeError("FAISS not installed. Vector search will not work.")

    def _initialize_database(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self._create_tables()
            logger.info("SQLite database initialized")
        except Exception as e:
            logger.error("Error initializing database: %s", e)
            raise RuntimeError(f"Failed to initialize database: {e}") from e

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                extraction_date TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                faiss_id INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                total_vectors INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)")
        self.conn.commit()
        logger.info("Database tables created/verified")

    def _load_or_create_index(self) -> None:
        if self.index_path.exists():
            self._load_index()
            self._verify_index_consistency()
        else:
            self._create_index()

    def _verify_index_consistency(self) -> None:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            db_chunk_count = cursor.fetchone()["count"]
            index_vector_count = self.index.ntotal if self.index else 0
            logger.debug(
                "Database chunks: %s, FAISS vectors: %s",
                db_chunk_count,
                index_vector_count,
            )
            if db_chunk_count == 0 and index_vector_count > 0:
                logger.warning("FAISS index exists but database is empty - clearing stale index")
                self._create_index()
            elif db_chunk_count > 0 and index_vector_count == 0:
                logger.warning("Database has chunks but FAISS index is empty - needs rebuild")
            elif abs(db_chunk_count - len(self.index_id_map)) > db_chunk_count * 0.1:
                logger.warning(
                    "Significant mismatch between database (%s) and index mapping (%s)",
                    db_chunk_count,
                    len(self.index_id_map),
                )
        except Exception as e:
            logger.error("Error verifying index consistency: %s", e)

    def _create_index(self) -> None:
        if self.embedding_dim is None:
            cursor = self.conn.cursor()
            cursor.execute("SELECT embedding_dim FROM index_metadata ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                self.embedding_dim = row["embedding_dim"]
            else:
                self.embedding_dim = DEFAULT_EMBEDDING_DIM
        try:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index_id_map = {}
            logger.info("Created new FAISS index with dimension %s", self.embedding_dim)
        except Exception as e:
            logger.error("Error creating FAISS index: %s", e)
            raise RuntimeError(f"Failed to create FAISS index: {e}") from e

    def _load_index(self) -> None:
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.embedding_dim = self.index.d
            self._rebuild_index_map()
            logger.info(
                "Loaded FAISS index: %s vectors, dim=%s",
                self.index.ntotal,
                self.embedding_dim,
            )
        except Exception as e:
            logger.error("Error loading FAISS index: %s", e)
            self._create_index()

    def _rebuild_index_map(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, faiss_id FROM chunks WHERE faiss_id IS NOT NULL ORDER BY faiss_id")
        self.index_id_map = {}
        for row in cursor.fetchall():
            self.index_id_map[row["faiss_id"]] = row["id"]
        logger.debug("Rebuilt index map with %s entries", len(self.index_id_map))

    def _save_index(self) -> None:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            logger.debug("FAISS index saved to disk")
        except Exception as e:
            logger.error("Error saving FAISS index: %s", e)

    def add_document(
        self,
        document_data: Dict[str, Any],
        chunks_data: List[Dict[str, Any]],
    ) -> int:
        if not chunks_data:
            raise ValueError("No chunks provided")
        logger.info(
            "Adding document: %s with %s chunks",
            document_data.get("file_name"),
            len(chunks_data),
        )
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (document_data["file_hash"],))
            existing = cursor.fetchone()
            if existing:
                logger.warning("Document already exists with hash %s", document_data["file_hash"])
                return existing["id"]

            cursor.execute("""
                INSERT INTO documents (
                    file_path, file_name, file_type, file_size, file_hash,
                    extraction_date, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                document_data["file_path"],
                document_data["file_name"],
                document_data["file_type"],
                document_data["file_size"],
                document_data["file_hash"],
                document_data["extraction_date"],
                json.dumps(document_data.get("metadata", {})),
            ))
            document_id = cursor.lastrowid
            embeddings = []
            chunk_ids = []

            for chunk_data in chunks_data:
                cursor.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, text, start_char, end_char,
                        token_count, embedding_model, embedding_dim
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    chunk_data["chunk_index"],
                    chunk_data["text"],
                    chunk_data["start_char"],
                    chunk_data["end_char"],
                    chunk_data["token_count"],
                    chunk_data["embedding_model"],
                    chunk_data["embedding_dim"],
                ))
                chunk_id = cursor.lastrowid
                chunk_ids.append(chunk_id)
                embeddings.append(chunk_data["embedding"])

            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings_array)
                start_faiss_id = self.index.ntotal
                self.index.add(embeddings_array)
                for i, chunk_id in enumerate(chunk_ids):
                    faiss_id = start_faiss_id + i
                    cursor.execute("UPDATE chunks SET faiss_id = ? WHERE id = ?", (faiss_id, chunk_id))
                    self.index_id_map[faiss_id] = chunk_id

            self._update_index_metadata(chunks_data[0]["embedding_model"], chunks_data[0]["embedding_dim"])
            self.conn.commit()
            self._save_index()
            logger.info("Added document %s with %s chunks", document_id, len(chunks_data))
            return document_id
        except Exception as e:
            logger.error("Error adding document: %s", e)
            self.conn.rollback()
            raise RuntimeError(f"Failed to add document: {e}") from e

    def _update_index_metadata(self, embedding_model: str, embedding_dim: int) -> None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM index_metadata ORDER BY id DESC LIMIT 1")
        existing = cursor.fetchone()
        if existing:
            cursor.execute("""
                UPDATE index_metadata
                SET total_vectors = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (self.index.ntotal, existing["id"]))
        else:
            cursor.execute("""
                INSERT INTO index_metadata (embedding_model, embedding_dim, total_vectors)
                VALUES (?, ?, ?)
            """, (embedding_model, embedding_dim, self.index.ntotal))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No vectors in index")
            return []
        top_k = top_k or Config.TOP_K_RETRIEVAL
        logger.debug("Searching for top %s similar chunks", top_k)
        try:
            query_norm = query_embedding.copy().astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_norm)
            start_time = time.time()
            scores, faiss_ids = self.index.search(query_norm, top_k)
            logger.debug("FAISS search completed in %.3fs", time.time() - start_time)

            results = []
            cursor = self.conn.cursor()
            valid_results = 0
            for i, (score, faiss_id) in enumerate(zip(scores[0], faiss_ids[0])):
                if faiss_id == -1:
                    continue
                if score < min_score:
                    continue
                valid_results += 1
                chunk_id = self.index_id_map.get(faiss_id)
                if chunk_id is None:
                    logger.warning("Missing chunk ID for FAISS ID %s", faiss_id)
                    continue
                cursor.execute("""
                    SELECT c.*, d.file_name, d.file_path, d.metadata as doc_metadata
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id = ?
                """, (chunk_id,))
                row = cursor.fetchone()
                if row:
                    results.append({
                        "chunk_id": chunk_id,
                        "faiss_id": faiss_id,
                        "score": float(score),
                        "rank": i + 1,
                        "text": row["text"],
                        "start_char": row["start_char"],
                        "end_char": row["end_char"],
                        "token_count": row["token_count"],
                        "chunk_index": row["chunk_index"],
                        "document_id": row["document_id"],
                        "file_name": row["file_name"],
                        "file_path": row["file_path"],
                        "document_metadata": json.loads(row["doc_metadata"] or "{}"),
                    })
            logger.info(
                "Vector search: %s valid results, %s returned after DB lookup",
                valid_results,
                len(results),
            )
            if len(results) == 0 and valid_results > 0:
                logger.warning("Valid FAISS results but no DB entries - attempting rebuild")
                if self._rebuild_faiss_index():
                    return self.search(query_embedding, top_k, min_score)
            return results
        except Exception as e:
            logger.error("Error searching vector store: %s", e)
            raise RuntimeError(f"Search failed: {e}") from e

    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.*, d.file_name, d.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ?
            ORDER BY c.chunk_index
        """, (document_id,))
        return [
            {
                "chunk_id": row["id"],
                "text": row["text"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
                "token_count": row["token_count"],
                "chunk_index": row["chunk_index"],
                "document_id": row["document_id"],
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "faiss_id": row["faiss_id"],
            }
            for row in cursor.fetchall()
        ]

    def list_documents(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.*, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """)
        return [
            {
                "document_id": row["id"],
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "file_type": row["file_type"],
                "file_size": row["file_size"],
                "file_hash": row["file_hash"],
                "extraction_date": row["extraction_date"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "chunk_count": row["chunk_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in cursor.fetchall()
        ]

    def clear_all_documents(self, new_embedding_dim: int) -> None:
        """Remove all documents and chunks, clear index metadata, and create a new empty
        FAISS index with the given dimension. Use when switching embedding model."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM index_metadata")
            self.conn.commit()
            self.embedding_dim = new_embedding_dim
            self._create_index()
            self.index_id_map = {}
            self._save_index()
            logger.info(
                "Cleared all documents and created new index with dimension %s",
                new_embedding_dim,
            )
        except Exception as e:
            logger.error("Error clearing documents: %s", e)
            self.conn.rollback()
            raise RuntimeError(f"Failed to clear documents: {e}") from e

    def delete_document(self, document_id: int) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_name FROM documents WHERE id = ?", (document_id,))
            doc_row = cursor.fetchone()
            if not doc_row:
                logger.warning("Document %s not found", document_id)
                return False
            cursor.execute(
                "SELECT faiss_id FROM chunks WHERE document_id = ? AND faiss_id IS NOT NULL",
                (document_id,),
            )
            faiss_ids = [row["faiss_id"] for row in cursor.fetchall()]
            cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE document_id = ?", (document_id,))
            chunk_count = cursor.fetchone()["count"]
            logger.info(
                "Deleting document %s (%s) with %s chunks and %s FAISS vectors",
                document_id,
                doc_row["file_name"],
                chunk_count,
                len(faiss_ids),
            )
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE document_id = ?", (document_id,))
                remaining = cursor.fetchone()["count"]
                if remaining > 0:
                    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                for faiss_id in faiss_ids:
                    self.index_id_map.pop(faiss_id, None)
                self.conn.commit()
                self._save_index()
                logger.info("Successfully deleted document %s", document_id)
                return True
            return False
        except Exception as e:
            logger.error("Error deleting document %s: %s", document_id, e)
            self.conn.rollback()
            return False

    def get_index_embedding_model(self) -> Optional[str]:
        """Return the embedding model stored for the current index, or None if empty."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT embedding_model FROM index_metadata ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row and row["embedding_model"]:
            return row["embedding_model"]
        cursor.execute(
            "SELECT embedding_model FROM chunks LIMIT 1"
        )
        row = cursor.fetchone()
        if row and row["embedding_model"]:
            return row["embedding_model"]
        return None

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        doc_count = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        chunk_count = cursor.fetchone()["count"]
        index_size = self.index.ntotal if self.index else 0
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        index_size_bytes = self.index_path.stat().st_size if self.index_path.exists() else 0
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "vectors_in_index": index_size,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.get_index_embedding_model(),
            "database_size_bytes": db_size,
            "index_size_bytes": index_size_bytes,
            "total_size_bytes": db_size + index_size_bytes,
        }

    def _rebuild_faiss_index(self) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT c.id, c.text, c.embedding_model, c.embedding_dim, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.id
            """)
            chunks_data = cursor.fetchall()
            if not chunks_data:
                self._create_index()
                return True
            logger.warning("Cannot rebuild index without stored embeddings. Creating fresh index.")
            self._create_index()
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM chunks")
            self.conn.commit()
            return False
        except Exception as e:
            logger.error("Error rebuilding FAISS index: %s", e)
            return False

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.index:
            self._save_index()
        logger.info("Vector store closed")


def create_vector_store(embedding_dim: Optional[int] = None) -> VectorStore:
    return VectorStore(embedding_dim=embedding_dim)
