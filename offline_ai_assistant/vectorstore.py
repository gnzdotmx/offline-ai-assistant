"""
Vector store module using FAISS for vector search and SQLite for metadata.

This module provides functionality to store, index, and search text embeddings
using FAISS for efficient similarity search and SQLite for metadata persistence.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import time
from datetime import datetime

try:
    import faiss
except ImportError:
    faiss = None

from .config import Config
from .embedder import TextEmbedder

logger = logging.getLogger("OfflineAIAssistant.vectorstore")


class VectorStore:
    """Vector store combining FAISS index with SQLite metadata."""
    
    def __init__(
        self,
        index_path: Path = None,
        db_path: Path = None,
        embedding_dim: int = None
    ):
        """
        Initialize the vector store.
        
        Args:
            index_path: Path to FAISS index file
            db_path: Path to SQLite database file
            embedding_dim: Dimension of embeddings
        """
        self.index_path = index_path or Config.FAISS_INDEX_PATH
        self.db_path = db_path or Config.SQLITE_DB_PATH
        self.embedding_dim = embedding_dim
        
        # FAISS index
        self.index = None
        self.index_id_map = {}  # Maps FAISS index positions to document IDs
        
        # SQLite connection
        self.conn = None
        
        # Initialize
        self._check_dependencies()
        self._initialize_database()
        self._load_or_create_index()
        
        logger.info(f"VectorStore initialized: index={self.index_path}, db={self.db_path}")
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if faiss is None:
            raise RuntimeError("FAISS not installed. Vector search will not work.")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Enable foreign key constraints (required for CASCADE DELETE)
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tables
            self._create_tables()
            
            logger.info("SQLite database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise RuntimeError(f"Failed to initialize database: {e}")
    
    def _create_tables(self) -> None:
        """Create required database tables."""
        cursor = self.conn.cursor()
        
        # Documents table
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
        
        # Chunks table
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
        
        # Index metadata table
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
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)")
        
        self.conn.commit()
        logger.info("Database tables created/verified")
    
    def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        if self.index_path.exists():
            self._load_index()
            # Check for database/index mismatch
            self._verify_index_consistency()
        else:
            self._create_index()
    
    def _verify_index_consistency(self) -> None:
        """Verify that FAISS index is consistent with database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            db_chunk_count = cursor.fetchone()['count']
            
            index_vector_count = self.index.ntotal if self.index else 0
            
            logger.debug(f"Database chunks: {db_chunk_count}, FAISS vectors: {index_vector_count}")
            
            if db_chunk_count == 0 and index_vector_count > 0:
                logger.warning("FAISS index exists but database is empty - clearing stale index")
                self._create_index()  # Create fresh empty index
                
            elif db_chunk_count > 0 and index_vector_count == 0:
                logger.warning("Database has chunks but FAISS index is empty - needs rebuild")
                
            elif abs(db_chunk_count - len(self.index_id_map)) > db_chunk_count * 0.1:  # More than 10% mismatch
                logger.warning(f"Significant mismatch between database ({db_chunk_count}) and index mapping ({len(self.index_id_map)})")
                
        except Exception as e:
            logger.error(f"Error verifying index consistency: {e}")
    
    def _create_index(self) -> None:
        """Create a new FAISS index."""
        if self.embedding_dim is None:
            # Try to get embedding dimension from database
            cursor = self.conn.cursor()
            cursor.execute("SELECT embedding_dim FROM index_metadata ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                self.embedding_dim = row['embedding_dim']
            else:
                # Default dimension for all-MiniLM-L6-v2
                self.embedding_dim = 384
        
        try:
            # Create FAISS index (using IndexFlatIP for cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index_id_map = {}
            
            logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise RuntimeError(f"Failed to create FAISS index: {e}")
    
    def _load_index(self) -> None:
        """Load existing FAISS index from disk."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.embedding_dim = self.index.d
            
            # Rebuild index ID map from database
            self._rebuild_index_map()
            
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors, dim={self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            # Create new index if loading fails
            self._create_index()
    
    def _rebuild_index_map(self) -> None:
        """Rebuild the index ID map from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, faiss_id FROM chunks WHERE faiss_id IS NOT NULL ORDER BY faiss_id")
        
        self.index_id_map = {}
        for row in cursor.fetchall():
            chunk_id = row['id']
            faiss_id = row['faiss_id']
            self.index_id_map[faiss_id] = chunk_id
        
        logger.debug(f"Rebuilt index map with {len(self.index_id_map)} entries")
    
    def _save_index(self) -> None:
        """Save FAISS index to disk."""
        try:
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(self.index, str(self.index_path))
            logger.debug("FAISS index saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def add_document(
        self,
        document_data: Dict[str, Any],
        chunks_data: List[Dict[str, Any]]
    ) -> int:
        """
        Add a document and its chunks to the vector store.
        
        Args:
            document_data: Document metadata
            chunks_data: List of chunk data with embeddings
            
        Returns:
            Document ID
        """
        if not chunks_data:
            raise ValueError("No chunks provided")
        
        logger.info(f"Adding document: {document_data.get('file_name')} with {len(chunks_data)} chunks")
        
        try:
            cursor = self.conn.cursor()
            
            # Check if document already exists
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (document_data['file_hash'],))
            existing = cursor.fetchone()
            if existing:
                logger.warning(f"Document already exists with hash {document_data['file_hash']}")
                return existing['id']
            
            # Insert document
            cursor.execute("""
                INSERT INTO documents (
                    file_path, file_name, file_type, file_size, file_hash,
                    extraction_date, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                document_data['file_path'],
                document_data['file_name'],
                document_data['file_type'],
                document_data['file_size'],
                document_data['file_hash'],
                document_data['extraction_date'],
                json.dumps(document_data.get('metadata', {}))
            ))
            
            document_id = cursor.lastrowid
            
            # Add chunks to FAISS and database
            embeddings = []
            chunk_ids = []
            
            for chunk_data in chunks_data:
                # Insert chunk into database
                cursor.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, text, start_char, end_char,
                        token_count, embedding_model, embedding_dim
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    chunk_data['chunk_index'],
                    chunk_data['text'],
                    chunk_data['start_char'],
                    chunk_data['end_char'],
                    chunk_data['token_count'],
                    chunk_data['embedding_model'],
                    chunk_data['embedding_dim']
                ))
                
                chunk_id = cursor.lastrowid
                chunk_ids.append(chunk_id)
                embeddings.append(chunk_data['embedding'])
            
            # Add embeddings to FAISS index
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings_array)
                
                # Get starting FAISS ID
                start_faiss_id = self.index.ntotal
                
                # Add to FAISS index
                self.index.add(embeddings_array)
                
                # Update database with FAISS IDs
                for i, chunk_id in enumerate(chunk_ids):
                    faiss_id = start_faiss_id + i
                    cursor.execute(
                        "UPDATE chunks SET faiss_id = ? WHERE id = ?",
                        (faiss_id, chunk_id)
                    )
                    self.index_id_map[faiss_id] = chunk_id
            
            # Update index metadata
            self._update_index_metadata(chunks_data[0]['embedding_model'], chunks_data[0]['embedding_dim'])
            
            self.conn.commit()
            self._save_index()
            
            logger.info(f"Added document {document_id} with {len(chunks_data)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            self.conn.rollback()
            raise RuntimeError(f"Failed to add document: {e}")
    
    def _update_index_metadata(self, embedding_model: str, embedding_dim: int) -> None:
        """Update index metadata."""
        cursor = self.conn.cursor()
        
        # Check if metadata exists
        cursor.execute("SELECT id FROM index_metadata ORDER BY id DESC LIMIT 1")
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute("""
                UPDATE index_metadata 
                SET total_vectors = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (self.index.ntotal, existing['id']))
        else:
            cursor.execute("""
                INSERT INTO index_metadata (embedding_model, embedding_dim, total_vectors)
                VALUES (?, ?, ?)
            """, (embedding_model, embedding_dim, self.index.ntotal))
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No vectors in index")
            return []
        
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        logger.debug(f"Searching for top {top_k} similar chunks")
        
        try:
            # Normalize query embedding
            query_norm = query_embedding.copy().astype(np.float32)
            query_norm = query_norm.reshape(1, -1)
            faiss.normalize_L2(query_norm)
            
            # Search FAISS index
            start_time = time.time()
            scores, faiss_ids = self.index.search(query_norm, top_k)
            search_time = time.time() - start_time
            
            logger.debug(f"FAISS search completed in {search_time:.3f}s")
            logger.debug(f"Raw scores: {scores[0][:5]}")  # Show first 5 scores
            logger.debug(f"Min score threshold: {min_score}")
            
            # Convert results
            results = []
            cursor = self.conn.cursor()
            
            valid_results = 0
            for i, (score, faiss_id) in enumerate(zip(scores[0], faiss_ids[0])):
                if faiss_id == -1:
                    logger.debug(f"Result {i}: faiss_id=-1 (no match)")
                    continue
                    
                if score < min_score:
                    logger.debug(f"Result {i}: score={score:.4f} < min_score={min_score} (filtered out)")
                    continue
                    
                valid_results += 1
                
                chunk_id = self.index_id_map.get(faiss_id)
                if chunk_id is None:
                    logger.warning(f"Missing chunk ID for FAISS ID {faiss_id}")
                    continue
                
                # Get chunk and document data
                cursor.execute("""
                    SELECT c.*, d.file_name, d.file_path, d.metadata as doc_metadata
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    result = {
                        "chunk_id": chunk_id,
                        "faiss_id": faiss_id,
                        "score": float(score),
                        "rank": i + 1,
                        "text": row['text'],
                        "start_char": row['start_char'],
                        "end_char": row['end_char'],
                        "token_count": row['token_count'],
                        "chunk_index": row['chunk_index'],
                        "document_id": row['document_id'],
                        "file_name": row['file_name'],
                        "file_path": row['file_path'],
                        "document_metadata": json.loads(row['doc_metadata'] or '{}')
                    }
                    results.append(result)
            
            logger.info(f"Vector search: {valid_results} valid results, {len(results)} returned after DB lookup")
            if len(results) == 0 and valid_results > 0:
                logger.warning("Valid FAISS results found but no database entries - possible index/DB mismatch")
                logger.info("Attempting to rebuild FAISS index to fix mismatch...")
                if self._rebuild_faiss_index():
                    logger.info("FAISS index rebuilt successfully, retrying search...")
                    # Retry the search with rebuilt index
                    return self.search(query_embedding, top_k, min_score)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk data
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.*, d.file_name, d.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ?
            ORDER BY c.chunk_index
        """, (document_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunk = {
                "chunk_id": row['id'],
                "text": row['text'],
                "start_char": row['start_char'],
                "end_char": row['end_char'],
                "token_count": row['token_count'],
                "chunk_index": row['chunk_index'],
                "document_id": row['document_id'],
                "file_name": row['file_name'],
                "file_path": row['file_path'],
                "faiss_id": row['faiss_id']
            }
            chunks.append(chunk)
        
        return chunks
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the store.
        
        Returns:
            List of document metadata
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.*, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """)
        
        documents = []
        for row in cursor.fetchall():
            doc = {
                "document_id": row['id'],
                "file_name": row['file_name'],
                "file_path": row['file_path'],
                "file_type": row['file_type'],
                "file_size": row['file_size'],
                "file_hash": row['file_hash'],
                "extraction_date": row['extraction_date'],
                "metadata": json.loads(row['metadata'] or '{}'),
                "chunk_count": row['chunk_count'],
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            }
            documents.append(doc)
        
        return documents
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and its chunks from the store.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Get document info for logging
            cursor.execute("SELECT file_name FROM documents WHERE id = ?", (document_id,))
            doc_row = cursor.fetchone()
            if not doc_row:
                logger.warning(f"Document {document_id} not found")
                return False
            
            file_name = doc_row['file_name']
            
            # Get FAISS IDs to remove from index
            cursor.execute("SELECT faiss_id FROM chunks WHERE document_id = ? AND faiss_id IS NOT NULL", (document_id,))
            faiss_ids = [row['faiss_id'] for row in cursor.fetchall()]
            
            # Count chunks before deletion
            cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE document_id = ?", (document_id,))
            chunk_count = cursor.fetchone()['count']
            
            logger.info(f"Deleting document {document_id} ({file_name}) with {chunk_count} chunks and {len(faiss_ids)} FAISS vectors")
            
            # Delete from database (should cascade to chunks due to foreign key constraint)
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                # Verify chunks were deleted (should be automatic with CASCADE)
                cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE document_id = ?", (document_id,))
                remaining_chunks = cursor.fetchone()['count']
                
                if remaining_chunks > 0:
                    logger.warning(f"Foreign key cascade failed - manually deleting {remaining_chunks} orphaned chunks")
                    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                
                # Remove from index ID map
                removed_count = 0
                for faiss_id in faiss_ids:
                    if faiss_id in self.index_id_map:
                        self.index_id_map.pop(faiss_id)
                        removed_count += 1
                
                # Note: FAISS doesn't support efficient deletion, so we keep the vectors
                # but they won't be returned in search results due to missing ID map entries
                
                self.conn.commit()
                self._save_index()
                
                logger.info(f"Successfully deleted document {document_id}: removed {removed_count} FAISS mappings")
                return True
            else:
                logger.warning(f"No document found with ID {document_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.conn.rollback()
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        cursor = self.conn.cursor()
        
        # Document count
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        doc_count = cursor.fetchone()['count']
        
        # Chunk count
        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        chunk_count = cursor.fetchone()['count']
        
        # Index stats
        index_size = self.index.ntotal if self.index else 0
        
        # Database size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        # Index file size
        index_size_bytes = self.index_path.stat().st_size if self.index_path.exists() else 0
        
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "vectors_in_index": index_size,
            "embedding_dim": self.embedding_dim,
            "database_size_bytes": db_size,
            "index_size_bytes": index_size_bytes,
            "total_size_bytes": db_size + index_size_bytes
        }
    
    def _rebuild_faiss_index(self) -> bool:
        """
        Rebuild the FAISS index from database embeddings.
        
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Get all chunks with embeddings (we need to regenerate embeddings)
            cursor.execute("""
                SELECT c.id, c.text, c.embedding_model, c.embedding_dim, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.id
            """)
            
            chunks_data = cursor.fetchall()
            if not chunks_data:
                logger.info("No chunks found in database, creating empty index")
                self._create_index()
                return True
            
            logger.info(f"Rebuilding FAISS index for {len(chunks_data)} chunks...")
            
            # We need to regenerate embeddings since we don't store them in the database
            # For now, create a fresh index and let the user re-upload documents
            logger.warning("Cannot rebuild index without stored embeddings. Creating fresh index.")
            logger.warning("Please re-upload your documents to rebuild the vector index.")
            
            # Create fresh index
            self._create_index()
            
            # Clear the database to force re-upload
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM chunks") 
            self.conn.commit()
            
            return False  # Return False to indicate user needs to re-upload
            
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
            return False
    
    def close(self) -> None:
        """Close database connection and save index."""
        if self.conn:
            self.conn.close()
            self.conn = None
        
        if self.index:
            self._save_index()
        
        logger.info("Vector store closed")


def create_vector_store(embedding_dim: int = None) -> VectorStore:
    """
    Convenience function to create a vector store.
    
    Args:
        embedding_dim: Dimension of embeddings
        
    Returns:
        VectorStore instance
    """
    return VectorStore(embedding_dim=embedding_dim)
