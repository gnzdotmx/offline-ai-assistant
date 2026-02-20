"""
RAG (Retrieval-Augmented Generation) orchestration.

Coordinates document processing, vector search, and LLM generation.
Uses config, data layer, and llm layer; depends on core.models for result types.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..config import Config
from ..data.extractor import DocumentExtractor
from ..data.chunker import TextChunker
from ..data.embedder import TextEmbedder
from ..data.vectorstore import VectorStore
from ..llm import LocalLLM

from .models import RAGResult, ProcessingResult, GenerationConfig
from . import rerank as rerank_module

logger = logging.getLogger("OfflineAIAssistant.rag")


def _cap_chunks_per_document(
    search_results: List[Dict[str, Any]],
    top_k: int,
    max_per_doc: int,
) -> List[Dict[str, Any]]:
    """
    Keep at most max_per_doc chunks per document_id, in retrieval order,
    until we have top_k chunks total. Fills remaining slots with next-best
    chunks from other documents.
    """
    if max_per_doc <= 0 or not search_results:
        return search_results
    capped: List[Dict[str, Any]] = []
    doc_counts: Dict[int, int] = {}
    for r in search_results:
        if len(capped) >= top_k:
            break
        doc_id = r.get("document_id")
        if doc_id is None:
            doc_id = 0
        if doc_counts.get(doc_id, 0) < max_per_doc:
            capped.append(r)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    return capped


def _apply_retrieval(
    query: str,
    query_embedding: Any,
    vector_store: VectorStore,
    effective_top_k: int,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Run vector search and optionally re-rank. Returns list of chunk dicts
    (up to effective_top_k) with rank/score set for downstream use.
    Optionally caps chunks per document (RAG_MAX_CHUNKS_PER_DOC) to diversify.
    """
    if Config.RAG_RERANK:
        candidate_k = min(
            effective_top_k * Config.RAG_RERANK_CANDIDATE_MULTIPLIER,
            50,
        )
        search_results = vector_store.search(
            query_embedding, top_k=candidate_k, min_score=min_score
        )
        search_results = rerank_module.rerank(query, search_results, effective_top_k)
    else:
        search_results = vector_store.search(
            query_embedding, top_k=effective_top_k, min_score=min_score
        )
    # Optional per-document cap: at most N chunks per document in final list
    if Config.RAG_MAX_CHUNKS_PER_DOC > 0:
        search_results = _cap_chunks_per_document(
            search_results, effective_top_k, Config.RAG_MAX_CHUNKS_PER_DOC
        )
    # Ensure rank reflects final order (1-based)
    for i, r in enumerate(search_results):
        r["rank"] = i + 1
        if "rerank_score" in r and Config.RAG_RERANK:
            r["score"] = r["rerank_score"]
    return search_results


def _reorder_results_by_document(
    search_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reorder search results for context: group by document_id, sort by chunk_index
    within each document, then order document groups by best retrieval rank
    (earliest rank in that doc). Sources keep original rank/score for UI.
    """
    if not search_results:
        return []
    # Group by document_id
    by_doc: Dict[int, List[Dict[str, Any]]] = {}
    for r in search_results:
        doc_id = r.get("document_id")
        if doc_id is None:
            doc_id = 0
        by_doc.setdefault(doc_id, []).append(r)
    # Sort chunks within each doc by chunk_index
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda x: x.get("chunk_index", 0))
    # Order document groups by best rank (min rank in that doc)
    def best_rank(items: List[Dict[str, Any]]) -> int:
        return min(it.get("rank", 999) for it in items)

    doc_ids_ordered = sorted(by_doc.keys(), key=lambda d: best_rank(by_doc[d]))
    reordered = []
    for doc_id in doc_ids_ordered:
        reordered.extend(by_doc[doc_id])
    return reordered


class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[LocalLLM] = None,
        extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[TextChunker] = None,
    ):
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore(
            embedding_dim=self.embedder.embedding_dim
        )
        self.llm = llm
        self.extractor = extractor or DocumentExtractor()
        self.chunker = chunker or TextChunker()
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "queries_answered": 0,
            "total_processing_time": 0.0,
            "total_query_time": 0.0,
        }
        logger.info("RAG pipeline initialized")

    def process_document(self, file_path: Path) -> ProcessingResult:
        start_time = time.time()
        logger.info("Processing document: %s", file_path)
        try:
            is_valid, error_msg = self.extractor.validate_file(file_path)
            if not is_valid:
                return ProcessingResult(
                    success=False,
                    document_id=None,
                    file_path=str(file_path),
                    chunks_created=0,
                    processing_time=time.time() - start_time,
                    error_message=error_msg,
                )
            document_data = self.extractor.extract_from_file(file_path)
            existing_docs = self.vector_store.list_documents()
            for doc in existing_docs:
                if doc["file_hash"] == document_data["file_hash"]:
                    logger.info("Document already processed: %s", file_path)
                    return ProcessingResult(
                        success=True,
                        document_id=doc["document_id"],
                        file_path=str(file_path),
                        chunks_created=doc["chunk_count"],
                        processing_time=time.time() - start_time,
                        error_message="Document already exists",
                    )
            managed_file_path = self._copy_document_to_storage(file_path)
            document_data["file_path"] = str(managed_file_path)
            chunks = self.chunker.chunk_text(
                document_data["full_text"],
                str(file_path),
                preserve_structure=True,
            )
            if not chunks:
                return ProcessingResult(
                    success=False,
                    document_id=None,
                    file_path=str(file_path),
                    chunks_created=0,
                    processing_time=time.time() - start_time,
                    error_message="No chunks created from document",
                )
            embedded_chunks = self.embedder.embed_chunks(
                chunks,
                batch_size=Config.EMBEDDING_BATCH_SIZE,
                show_progress=Config.EMBEDDING_SHOW_PROGRESS,
            )
            document_id = self.vector_store.add_document(document_data, embedded_chunks)
            processing_time = time.time() - start_time
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["total_processing_time"] += processing_time
            logger.info(
                "Document processed successfully: %s chunks in %.2fs",
                len(chunks),
                processing_time,
            )
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                chunks_created=len(chunks),
                processing_time=processing_time,
            )
        except Exception as e:
            logger.error("Error processing document %s: %s", file_path, e)
            return ProcessingResult(
                success=False,
                document_id=None,
                file_path=str(file_path),
                chunks_created=0,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    def query(
        self,
        query: str,
        template: str = "default",
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        generation_config: Optional[GenerationConfig] = None,
    ) -> RAGResult:
        start_time = time.time()
        logger.info("Processing query: %s...", query[:100])
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Cannot generate responses.")
        try:
            retrieval_start = time.time()
            query_embedding = self.embedder.embed_query(query)
            effective_top_k = top_k or Config.TOP_K_RETRIEVAL
            effective_min_score = (
                min_score if min_score > 0 else Config.MIN_SCORE_RETRIEVAL
            )
            search_results = _apply_retrieval(
                query,
                query_embedding,
                self.vector_store,
                effective_top_k,
                min_score=effective_min_score,
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
                    template_used=template,
                )
            # Optionally reorder by document for coherent context; sources keep retrieval rank/score
            if Config.RAG_CONTEXT_ORDER == "document_order":
                ordered_results = _reorder_results_by_document(search_results)
            else:
                ordered_results = search_results
            context_chunks = []
            sources = []
            for result in ordered_results:
                context_chunks.append(result["text"])
                sources.append({
                    "rank": result["rank"],
                    "score": result["score"],
                    "file_name": result["file_name"],
                    "file_path": result["file_path"],
                    "chunk_index": result["chunk_index"],
                    "start_char": result["start_char"],
                    "end_char": result["end_char"],
                    "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                })
            prompt_template = Config.PROMPT_TEMPLATES.get(template, Config.PROMPT_TEMPLATES["default"])
            rag_prompt = self.llm.create_rag_prompt(query, context_chunks, prompt_template)
            rag_prompt = self.llm.truncate_to_context(
                rag_prompt, max_tokens=int(self.llm.n_ctx * 0.8)
            )
            config = generation_config or GenerationConfig(
                max_tokens=Config.LLM_MAX_TOKENS,
                temperature=Config.LLM_TEMPERATURE,
                top_p=Config.LLM_TOP_P,
                stop_sequences=[
                    "Human:", "User:", "\n\nHuman:", "\n\nUser:",
                    "\n\nQuestion:", "\n\nContext:", "\n\n\n",
                    "In conclusion", "To summarize", "In summary",
                ],
            )
            generation_start = time.time()
            response_text = self.llm.generate_complete(rag_prompt, config)
            generation_time = time.time() - generation_start
            tokens_generated = self.llm.estimate_tokens(response_text)
            total_time = time.time() - start_time
            self.stats["queries_answered"] += 1
            self.stats["total_query_time"] += total_time
            logger.info(
                "Query answered in %.2fs (retrieval: %.2fs, generation: %.2fs)",
                total_time,
                retrieval_time,
                generation_time,
            )
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
                template_used=template,
            )
        except Exception as e:
            logger.error("Error processing query: %s", e)
            raise RuntimeError(f"Query processing failed: {e}") from e

    def query_stream(
        self,
        query: str,
        template: str = "default",
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Iterator[Dict[str, Any]]:
        start_time = time.time()
        logger.info("Processing streaming query: %s...", query[:100])
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Cannot generate responses.")
        if not self.llm.is_loaded():
            raise RuntimeError("LLM model not loaded. Cannot generate responses.")
        try:
            yield {"type": "status", "message": "Searching for relevant information..."}
            retrieval_start = time.time()
            query_embedding = self.embedder.embed_query(query)
            effective_top_k = top_k or Config.TOP_K_RETRIEVAL
            effective_min_score = (
                min_score if min_score > 0 else Config.MIN_SCORE_RETRIEVAL
            )
            search_results = _apply_retrieval(
                query,
                query_embedding,
                self.vector_store,
                effective_top_k,
                min_score=effective_min_score,
            )
            retrieval_time = time.time() - retrieval_start
            logger.info("Found %s relevant chunks in %.2fs", len(search_results), retrieval_time)
            if not search_results:
                yield {
                    "type": "final",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "retrieval_time": retrieval_time,
                    "generation_time": 0.0,
                    "total_time": time.time() - start_time,
                    "tokens_generated": 0,
                    "chunks_retrieved": 0,
                }
                return
            # Optionally reorder by document for coherent context; sources keep retrieval rank/score
            if Config.RAG_CONTEXT_ORDER == "document_order":
                ordered_results = _reorder_results_by_document(search_results)
            else:
                ordered_results = search_results
            sources = []
            context_chunks = []
            for result in ordered_results:
                context_chunks.append(result["text"])
                sources.append({
                    "rank": result["rank"],
                    "score": result["score"],
                    "file_name": result["file_name"],
                    "chunk_index": result["chunk_index"],
                    "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                })
            yield {"type": "sources", "sources": sources, "retrieval_time": retrieval_time}
            yield {"type": "status", "message": "Generating response..."}
            prompt_template = Config.PROMPT_TEMPLATES.get(template, Config.PROMPT_TEMPLATES["default"])
            rag_prompt = self.llm.create_rag_prompt(query, context_chunks, prompt_template)
            rag_prompt = self.llm.truncate_to_context(
                rag_prompt, max_tokens=int(self.llm.n_ctx * 0.8)
            )
            config = generation_config or GenerationConfig(
                max_tokens=Config.LLM_MAX_TOKENS,
                temperature=Config.LLM_TEMPERATURE,
                top_p=Config.LLM_TOP_P,
                stop_sequences=[
                    "Human:", "User:", "\n\nHuman:", "\n\nUser:",
                    "\n\nQuestion:", "\n\nContext:", "\n\n\n",
                    "In conclusion", "To summarize", "In summary",
                ],
                stream=True,
            )
            generation_start = time.time()
            full_response = ""
            token_count = 0
            for token in self.llm.generate(rag_prompt, config):
                full_response += token
                token_count += 1
                yield {"type": "token", "token": token, "partial_answer": full_response}
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            self.stats["queries_answered"] += 1
            self.stats["total_query_time"] += total_time
            yield {
                "type": "final",
                "answer": full_response.strip(),
                "sources": sources,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "tokens_generated": self.llm.estimate_tokens(full_response),
                "chunks_retrieved": len(search_results),
            }
        except Exception as e:
            logger.error("Error in streaming query: %s", e)
            yield {"type": "error", "error": str(e)}

    def generate_content(
        self,
        template: str,
        context_query: Optional[str] = None,
        direct_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> RAGResult:
        start_time = time.time()
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Cannot generate content.")
        logger.info("Generating content with template: %s", template)
        try:
            sources = []
            retrieval_time = 0.0
            if context_query:
                retrieval_start = time.time()
                query_embedding = self.embedder.embed_query(context_query)
                search_results = _apply_retrieval(
                    context_query,
                    query_embedding,
                    self.vector_store,
                    Config.TOP_K_RETRIEVAL,
                    min_score=Config.MIN_SCORE_RETRIEVAL,
                )
                retrieval_time = time.time() - retrieval_start
                if Config.RAG_CONTEXT_ORDER == "document_order":
                    ordered_results = _reorder_results_by_document(search_results)
                else:
                    ordered_results = search_results
                context_chunks = [result["text"] for result in ordered_results]
                sources = [
                    {
                        "rank": result["rank"],
                        "score": result["score"],
                        "file_name": result["file_name"],
                        "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                    }
                    for result in ordered_results
                ]
                prompt_template = Config.PROMPT_TEMPLATES.get(template, Config.PROMPT_TEMPLATES["default"])
                prompt = prompt_template.format(
                    context="\n\n".join(context_chunks),
                    question=context_query,
                )
            elif direct_prompt:
                prompt = direct_prompt
            else:
                raise ValueError("Either context_query or direct_prompt must be provided")
            generation_start = time.time()
            config = generation_config or GenerationConfig(
                max_tokens=Config.LLM_MAX_TOKENS * 2,
                temperature=Config.LLM_TEMPERATURE,
                top_p=Config.LLM_TOP_P,
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
                template_used=template,
            )
        except Exception as e:
            logger.error("Error generating content: %s", e)
            raise RuntimeError(f"Content generation failed: {e}") from e

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.vector_store.list_documents()

    def delete_document(self, document_id: int) -> bool:
        logger.info("Deleting document: %s", document_id)
        try:
            docs = self.vector_store.list_documents()
            doc_info = None
            for doc in docs:
                if doc["document_id"] == document_id:
                    doc_info = doc
                    break
            success = self.vector_store.delete_document(document_id)
            if success:
                self.stats["documents_processed"] = max(0, self.stats["documents_processed"] - 1)
                if doc_info:
                    file_path = Path(doc_info["file_path"])
                    if file_path.exists() and file_path.parent == Config.DOCS_DIR:
                        try:
                            file_path.unlink()
                            logger.info("Deleted document file: %s", file_path)
                        except Exception as e:
                            logger.warning("Could not delete document file %s: %s", file_path, e)
            return success
        except Exception as e:
            logger.error("Error deleting document: %s", e)
            return False

    def _copy_document_to_storage(self, source_path: Path) -> Path:
        Config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        dest_filename = source_path.name
        dest_path = Config.DOCS_DIR / dest_filename
        counter = 1
        while dest_path.exists():
            try:
                if dest_path.resolve() == source_path.resolve():
                    logger.debug("Document already in storage: %s", dest_path)
                    return dest_path
            except OSError:
                pass
            stem, suffix = source_path.stem, source_path.suffix
            dest_filename = f"{stem}_{counter}{suffix}"
            dest_path = Config.DOCS_DIR / dest_filename
            counter += 1
        try:
            shutil.copy2(source_path, dest_path)
            logger.info("Copied document to storage: %s", dest_path)
            return dest_path
        except Exception as e:
            logger.error("Error copying document: %s", e)
            return source_path

    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        return self.vector_store.get_document_chunks(document_id)

    def get_statistics(self) -> Dict[str, Any]:
        vector_stats = self.vector_store.get_stats()
        llm_info = self.llm.get_model_info() if self.llm else {"status": "not_loaded"}
        embedder_info = self.embedder.get_model_info()
        return {
            "pipeline_stats": self.stats,
            "vector_store": vector_stats,
            "llm": llm_info,
            "embedder": embedder_info,
            "avg_processing_time": self.stats["total_processing_time"] / max(1, self.stats["documents_processed"]),
            "avg_query_time": self.stats["total_query_time"] / max(1, self.stats["queries_answered"]),
        }

    def close(self) -> None:
        if self.vector_store:
            self.vector_store.close()
        if self.llm:
            self.llm.unload_model()
        logger.info("RAG pipeline closed")


def create_rag_pipeline(
    model_path: Optional[Path] = None,
    embedding_model: Optional[str] = None,
) -> RAGPipeline:
    embedder = TextEmbedder(model_name=embedding_model)
    vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
    llm = None
    if model_path and model_path.exists():
        llm = LocalLLM(model_path=model_path)
    return RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )
