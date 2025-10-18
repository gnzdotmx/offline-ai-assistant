"""
RAG (Retrieval-Augmented Generation) orchestration module.

This module provides the main RAG pipeline that coordinates document processing,
vector search, and LLM generation to provide AI-powered responses with citations.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple
import time
import shutil
from dataclasses import dataclass, asdict

from .config import Config
from .extractor import DocumentExtractor
from .chunker import TextChunker, TextChunk
from .embedder import TextEmbedder
from .vectorstore import VectorStore
from .llm import LocalLLM, GenerationConfig

logger = logging.getLogger("OfflineAIAssistant.rag")


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


class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(
        self,
        embedder: TextEmbedder = None,
        vector_store: VectorStore = None,
        llm: LocalLLM = None,
        extractor: DocumentExtractor = None,
        chunker: TextChunker = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedder: Text embedder instance
            vector_store: Vector store instance
            llm: Local LLM instance
            extractor: Document extractor instance
            chunker: Text chunker instance
        """
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore(
            embedding_dim=self.embedder.embedding_dim
        )
        self.llm = llm
        self.extractor = extractor or DocumentExtractor()
        self.chunker = chunker or TextChunker()
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "queries_answered": 0,
            "total_processing_time": 0.0,
            "total_query_time": 0.0
        }
        
        logger.info("RAG pipeline initialized")
    
    def process_document(self, file_path: Path) -> ProcessingResult:
        """
        Process a document through the complete RAG pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessingResult with processing information
        """
        start_time = time.time()
        
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Validate file
            is_valid, error_msg = self.extractor.validate_file(file_path)
            if not is_valid:
                return ProcessingResult(
                    success=False,
                    document_id=None,
                    file_path=str(file_path),
                    chunks_created=0,
                    processing_time=0.0,
                    error_message=error_msg
                )
            
            # Extract text and metadata
            logger.debug("Extracting text from document")
            document_data = self.extractor.extract_from_file(file_path)
            
            # Check if document already exists
            existing_docs = self.vector_store.list_documents()
            for doc in existing_docs:
                if doc['file_hash'] == document_data['file_hash']:
                    logger.info(f"Document already processed: {file_path}")
                    return ProcessingResult(
                        success=True,
                        document_id=doc['document_id'],
                        file_path=str(file_path),
                        chunks_created=doc['chunk_count'],
                        processing_time=time.time() - start_time,
                        error_message="Document already exists"
                    )
            
            # Copy document to managed docs directory
            logger.debug("Copying document to managed storage")
            managed_file_path = self._copy_document_to_storage(file_path)
            # Update document_data to use the managed path
            document_data['file_path'] = str(managed_file_path)
            
            # Chunk the text
            logger.debug("Chunking document text")
            chunks = self.chunker.chunk_text(
                document_data['full_text'],
                str(file_path),
                preserve_structure=True
            )
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    document_id=None,
                    file_path=str(file_path),
                    chunks_created=0,
                    processing_time=time.time() - start_time,
                    error_message="No chunks created from document"
                )
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(chunks)} chunks")
            embedded_chunks = self.embedder.embed_chunks(
                chunks,
                batch_size=32,
                show_progress=True
            )
            
            # Store in vector database
            logger.debug("Storing document and chunks in vector database")
            document_id = self.vector_store.add_document(
                document_data,
                embedded_chunks
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["total_processing_time"] += processing_time
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                chunks_created=len(chunks),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return ProcessingResult(
                success=False,
                document_id=None,
                file_path=str(file_path),
                chunks_created=0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def query(
        self,
        query: str,
        template: str = "default",
        top_k: int = None,
        min_score: float = 0.0,
        generation_config: GenerationConfig = None
    ) -> RAGResult:
        """
        Answer a query using RAG.
        
        Args:
            query: User query
            template: Prompt template to use
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score
            generation_config: LLM generation configuration
            
        Returns:
            RAGResult with answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {query[:100]}...")
        
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Cannot generate responses.")
        
        try:
            # Retrieve relevant chunks
            retrieval_start = time.time()
            
            logger.debug("Generating query embedding")
            query_embedding = self.embedder.embed_query(query)
            
            logger.debug("Searching vector store")
            search_results = self.vector_store.search(
                query_embedding,
                top_k=top_k or Config.TOP_K_RETRIEVAL,
                min_score=min_score
            )
            
            retrieval_time = time.time() - retrieval_start
            
            if not search_results:
                logger.warning("No relevant chunks found for query")
                return RAGResult(
                    query=query,
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    generation_time=0.0,
                    retrieval_time=retrieval_time,
                    total_time=time.time() - start_time,
                    tokens_generated=0,
                    chunks_retrieved=0,
                    model_used=self.llm.model_path.name if self.llm.model_path else "unknown",
                    template_used=template
                )
            
            # Prepare context and sources
            context_chunks = []
            sources = []
            
            for i, result in enumerate(search_results):
                # Add to context
                context_chunks.append(result['text'])
                
                # Prepare source information
                source = {
                    "rank": i + 1,
                    "score": result['score'],
                    "file_name": result['file_name'],
                    "file_path": result['file_path'],
                    "chunk_index": result['chunk_index'],
                    "start_char": result['start_char'],
                    "end_char": result['end_char'],
                    "text_preview": result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                }
                sources.append(source)
            
            # Create RAG prompt
            logger.debug(f"Creating RAG prompt with {len(context_chunks)} context chunks")
            prompt_template = Config.PROMPT_TEMPLATES.get(template, Config.PROMPT_TEMPLATES["default"])
            rag_prompt = self.llm.create_rag_prompt(
                query,
                context_chunks,
                prompt_template
            )
            
            # Truncate prompt if necessary
            rag_prompt = self.llm.truncate_to_context(
                rag_prompt,
                max_tokens=int(self.llm.n_ctx * 0.8)  # Leave room for generation
            )

            # Generate response
            generation_start = time.time()
            logger.debug("Generating LLM response")
            
            config = generation_config or GenerationConfig(
                max_tokens=Config.LLM_MAX_TOKENS,
                temperature=Config.LLM_TEMPERATURE,
                top_p=Config.LLM_TOP_P,
                stop_sequences=[
                    "Human:", "User:", "\n\nHuman:", "\n\nUser:",
                    "\n\nQuestion:", "\n\nContext:", "\n\n\n",
                    "In conclusion", "To summarize", "In summary"
                ]
            )
            
            response_text = self.llm.generate_complete(rag_prompt, config)
            generation_time = time.time() - generation_start
            
            # Count generated tokens
            tokens_generated = self.llm.estimate_tokens(response_text)
            
            total_time = time.time() - start_time
            
            # Update statistics
            self.stats["queries_answered"] += 1
            self.stats["total_query_time"] += total_time
            
            logger.info(f"Query answered in {total_time:.2f}s "
                       f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
            
            return RAGResult(
                query=query,
                answer=response_text.strip(),
                sources=sources,
                generation_time=generation_time,
                retrieval_time=retrieval_time,
                total_time=total_time,
                tokens_generated=tokens_generated,
                chunks_retrieved=len(search_results),
                model_used=self.llm.model_path.name if self.llm.model_path else "unknown",
                template_used=template
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise RuntimeError(f"Query processing failed: {e}")
    
    def query_stream(
        self,
        query: str,
        template: str = "default",
        top_k: int = None,
        min_score: float = 0.0,
        generation_config: GenerationConfig = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Answer a query using RAG with streaming response.
        
        Args:
            query: User query
            template: Prompt template to use
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score
            generation_config: LLM generation configuration
            
        Yields:
            Dictionary with streaming updates
        """
        start_time = time.time()
        
        logger.info(f"Processing streaming query: {query[:100]}...")
        
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Cannot generate responses.")
        
        if not self.llm.is_loaded():
            raise RuntimeError("LLM model not loaded. Cannot generate responses.")
        
        logger.debug(f"LLM model info: {self.llm.get_model_info()}")
        
        try:
            # Retrieve relevant chunks
            yield {"type": "status", "message": "Searching for relevant information..."}
            
            retrieval_start = time.time()
            logger.debug("Generating query embedding...")
            query_embedding = self.embedder.embed_query(query)
            
            logger.debug(f"Searching vector store with top_k={top_k or Config.TOP_K_RETRIEVAL}, min_score={min_score}...")
            # Temporarily allow all results for debugging if min_score is default
            effective_min_score = min_score if min_score > 0 else -1.0
            search_results = self.vector_store.search(
                query_embedding,
                top_k=top_k or Config.TOP_K_RETRIEVAL,
                min_score=effective_min_score
            )
            retrieval_time = time.time() - retrieval_start
            
            logger.info(f"Found {len(search_results)} relevant chunks in {retrieval_time:.2f}s")
            
            if not search_results:
                yield {
                    "type": "final",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "retrieval_time": retrieval_time,
                    "generation_time": 0.0,
                    "total_time": time.time() - start_time,
                    "tokens_generated": 0,
                    "chunks_retrieved": 0
                }
                return
            
            # Send sources
            sources = []
            context_chunks = []
            
            for i, result in enumerate(search_results):
                context_chunks.append(result['text'])
                source = {
                    "rank": i + 1,
                    "score": result['score'],
                    "file_name": result['file_name'],
                    "chunk_index": result['chunk_index'],
                    "text_preview": result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                }
                sources.append(source)
            
            yield {
                "type": "sources",
                "sources": sources,
                "retrieval_time": retrieval_time
            }
            
            # Create prompt and generate
            yield {"type": "status", "message": "Generating response..."}
            
            prompt_template = Config.PROMPT_TEMPLATES.get(template, Config.PROMPT_TEMPLATES["default"])
            rag_prompt = self.llm.create_rag_prompt(query, context_chunks, prompt_template)
            rag_prompt = self.llm.truncate_to_context(rag_prompt, max_tokens=int(self.llm.n_ctx * 0.8))
            
            config = generation_config or GenerationConfig(
                max_tokens=Config.LLM_MAX_TOKENS,
                temperature=Config.LLM_TEMPERATURE,
                top_p=Config.LLM_TOP_P,
                stop_sequences=[
                    "Human:", "User:", "\n\nHuman:", "\n\nUser:",
                    "\n\nQuestion:", "\n\nContext:", "\n\n\n",  # Stop at new sections
                    "In conclusion", "To summarize", "In summary"  # Stop at conclusion phrases
                ],
                stream=True
            )
            
            generation_start = time.time()
            full_response = ""
            token_count = 0
            
            logger.info("Starting LLM token generation...")
            
            # Stream tokens
            for token in self.llm.generate(rag_prompt, config):
                full_response += token
                token_count += 1
                
                if token_count % 10 == 0:  # Log every 10 tokens
                    logger.debug(f"Generated {token_count} tokens...")
                
                yield {
                    "type": "token",
                    "token": token,
                    "partial_answer": full_response
                }
            
            logger.info(f"LLM generation completed: {token_count} tokens generated")
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            # Final result
            yield {
                "type": "final",
                "answer": full_response.strip(),
                "sources": sources,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "tokens_generated": self.llm.estimate_tokens(full_response),
                "chunks_retrieved": len(search_results)
            }
            
            # Update statistics
            self.stats["queries_answered"] += 1
            self.stats["total_query_time"] += total_time
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def generate_content(
        self,
        template: str,
        context_query: str = None,
        direct_prompt: str = None,
        generation_config: GenerationConfig = None
    ) -> RAGResult:
        """
        Generate content using a specific template.
        
        Args:
            template: Template name to use
            context_query: Query to retrieve context (optional)
            direct_prompt: Direct prompt without retrieval (optional)
            generation_config: Generation configuration
            
        Returns:
            RAGResult with generated content
        """
        start_time = time.time()
        
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Cannot generate content.")
        
        logger.info(f"Generating content with template: {template}")
        
        try:
            sources = []
            retrieval_time = 0.0
            
            if context_query:
                # Retrieve context for content generation
                retrieval_start = time.time()
                query_embedding = self.embedder.embed_query(context_query)
                search_results = self.vector_store.search(
                    query_embedding,
                    top_k=Config.TOP_K_RETRIEVAL
                )
                retrieval_time = time.time() - retrieval_start
                
                context_chunks = [result['text'] for result in search_results]
                sources = [{
                    "rank": i + 1,
                    "score": result['score'],
                    "file_name": result['file_name'],
                    "text_preview": result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                } for i, result in enumerate(search_results)]
                
                # Create prompt with context
                prompt_template = Config.PROMPT_TEMPLATES.get(template, Config.PROMPT_TEMPLATES["default"])
                prompt = prompt_template.format(
                    context="\n\n".join(context_chunks),
                    question=context_query
                )
            elif direct_prompt:
                prompt = direct_prompt
            else:
                raise ValueError("Either context_query or direct_prompt must be provided")
            
            # Generate content
            generation_start = time.time()
            config = generation_config or GenerationConfig(
                max_tokens=Config.LLM_MAX_TOKENS * 2,  # Allow longer generation for content
                temperature=Config.LLM_TEMPERATURE,
                top_p=Config.LLM_TOP_P
            )
            
            response_text = self.llm.generate_complete(prompt, config)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            return RAGResult(
                query=context_query or "Content Generation",
                answer=response_text.strip(),
                sources=sources,
                generation_time=generation_time,
                retrieval_time=retrieval_time,
                total_time=total_time,
                tokens_generated=self.llm.estimate_tokens(response_text),
                chunks_retrieved=len(sources),
                model_used=self.llm.model_path.name if self.llm.model_path else "unknown",
                template_used=template
            )
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise RuntimeError(f"Content generation failed: {e}")
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all processed documents.
        
        Returns:
            List of document information
        """
        return self.vector_store.list_documents()
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document from the system.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        logger.info(f"Deleting document: {document_id}")
        
        # Get document info before deletion to delete the physical file
        try:
            docs = self.vector_store.list_documents()
            doc_info = None
            for doc in docs:
                if doc['document_id'] == document_id:
                    doc_info = doc
                    break
            
            # Delete from vector store
            success = self.vector_store.delete_document(document_id)
            
            if success:
                self.stats["documents_processed"] = max(0, self.stats["documents_processed"] - 1)
                
                # Try to delete the physical file if it's in our managed directory
                if doc_info:
                    file_path = Path(doc_info['file_path'])
                    if file_path.exists() and file_path.parent == Config.DOCS_DIR:
                        try:
                            file_path.unlink()
                            logger.info(f"Deleted document file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not delete document file {file_path}: {e}")
            
            return success
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def _copy_document_to_storage(self, source_path: Path) -> Path:
        """
        Copy a document to the managed docs directory.
        
        Args:
            source_path: Original document path
            
        Returns:
            Path to the copied document
        """
        # Ensure docs directory exists
        Config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Determine destination path
        dest_filename = source_path.name
        dest_path = Config.DOCS_DIR / dest_filename
        
        # Handle filename conflicts by adding a number suffix
        counter = 1
        while dest_path.exists():
            # Check if it's the same file (by comparing paths)
            try:
                if dest_path.resolve() == source_path.resolve():
                    logger.debug(f"Document already in storage: {dest_path}")
                    return dest_path
            except:
                pass
            
            # Add counter to filename
            stem = source_path.stem
            suffix = source_path.suffix
            dest_filename = f"{stem}_{counter}{suffix}"
            dest_path = Config.DOCS_DIR / dest_filename
            counter += 1
        
        # Copy the file
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied document to storage: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"Error copying document: {e}")
            # If copy fails, use original path
            return source_path
    
    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk information
        """
        return self.vector_store.get_document_chunks(document_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get RAG pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        vector_stats = self.vector_store.get_stats()
        llm_info = self.llm.get_model_info() if self.llm else {"status": "not_loaded"}
        embedder_info = self.embedder.get_model_info()
        
        return {
            "pipeline_stats": self.stats,
            "vector_store": vector_stats,
            "llm": llm_info,
            "embedder": embedder_info,
            "avg_processing_time": (
                self.stats["total_processing_time"] / max(1, self.stats["documents_processed"])
            ),
            "avg_query_time": (
                self.stats["total_query_time"] / max(1, self.stats["queries_answered"])
            )
        }
    
    def close(self) -> None:
        """Close all pipeline components."""
        if self.vector_store:
            self.vector_store.close()
        
        if self.llm:
            self.llm.unload_model()
        
        logger.info("RAG pipeline closed")


def create_rag_pipeline(
    model_path: Path = None,
    embedding_model: str = None
) -> RAGPipeline:
    """
    Convenience function to create a complete RAG pipeline.
    
    Args:
        model_path: Path to LLM model file
        embedding_model: Name of embedding model
        
    Returns:
        RAGPipeline instance
    """
    # Initialize components
    embedder = TextEmbedder(model_name=embedding_model)
    vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
    
    llm = None
    if model_path and model_path.exists():
        llm = LocalLLM(model_path=model_path)
    
    return RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm
    )
