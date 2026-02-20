"""
Configuration schema and validation.

Validates types and bounds for all user-editable settings to prevent
malicious or malformed config from causing unsafe behavior.

Chunking / encoding:
  encoding_model: Tiktoken encoding name (e.g. "cl100k_base"). Used for token counting
  and chunk boundaries. Ideally set to match the LLM's tokenizer when possible (e.g.
  if using a LLaMA-based model with a different tokenizer, use that encoding) so
  chunk boundaries align better with the model's context window.
  word_fallback_chunk_ratio: When tiktoken is unavailable, the chunker uses word-based
  sizing; this ratio (0.0--1.0) scales chunk_size so effective size = chunk_size * ratio
  (default 0.5) to keep chunks under the LLM context.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger("OfflineAIAssistant.config.schema")

# Bounds and constraints for config values (security and sanity)
# top_k_retrieval: recommended range 1–50; fewer for simple Q&A, more for complex/multi-doc.
# min_score_retrieval: similarity threshold [-1, 1] (cosine); use -1 to disable filtering.
CONFIG_BOUNDS = {
    "chunk_size": (64, 4096),
    "chunk_overlap": (0, 1024),
    "word_fallback_chunk_ratio": (0.25, 1.0),
    "top_k_retrieval": (1, 50),
    "min_score_retrieval": (-1.0, 1.0),
    "rag_rerank_candidate_multiplier": (2, 5),
    "rag_max_chunks_per_doc": (0, 50),  # 0 = no cap; max chunks per document in final retrieval list
    "llm_max_tokens": (1, 8192),
    "llm_temperature": (0.0, 2.0),
    "llm_top_p": (0.0, 1.0),
    "llm_n_gpu_layers": (-1, 1024),
    "llm_n_batch": (64, 2048),
    "embedding_batch_size": (1, 512),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def validate_settings(settings: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    """
    Validate and sanitize settings dict. Returns (validated_dict, list of warnings).
    """
    validated = {}
    warnings = []

    # chunk_size
    v = settings.get("chunk_size")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["chunk_size"]
            validated["chunk_size"] = int(_clamp(n, low, high))
            if n != validated["chunk_size"]:
                warnings.append(f"chunk_size clamped to {validated['chunk_size']}")
        except (TypeError, ValueError):
            warnings.append("Invalid chunk_size; using default")

    # chunk_overlap
    v = settings.get("chunk_overlap")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["chunk_overlap"]
            validated["chunk_overlap"] = int(_clamp(n, low, high))
            if n != validated["chunk_overlap"]:
                warnings.append(f"chunk_overlap clamped to {validated['chunk_overlap']}")
        except (TypeError, ValueError):
            warnings.append("Invalid chunk_overlap; using default")

    # top_k_retrieval
    v = settings.get("top_k_retrieval")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["top_k_retrieval"]
            validated["top_k_retrieval"] = int(_clamp(n, low, high))
        except (TypeError, ValueError):
            warnings.append("Invalid top_k_retrieval; using default")

    # min_score_retrieval: float in [-1, 1]; chunks below this similarity are excluded (-1 = no filter)
    v = settings.get("min_score_retrieval")
    if v is not None:
        try:
            x = float(v)
            low, high = CONFIG_BOUNDS["min_score_retrieval"]
            validated["min_score_retrieval"] = _clamp(x, low, high)
            if x != validated["min_score_retrieval"]:
                warnings.append(
                    f"min_score_retrieval clamped to {validated['min_score_retrieval']}"
                )
        except (TypeError, ValueError):
            warnings.append("Invalid min_score_retrieval; using default")

    # llm_max_tokens
    v = settings.get("llm_max_tokens")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["llm_max_tokens"]
            validated["llm_max_tokens"] = int(_clamp(n, low, high))
        except (TypeError, ValueError):
            warnings.append("Invalid llm_max_tokens; using default")

    # llm_temperature
    v = settings.get("llm_temperature")
    if v is not None:
        try:
            x = float(v)
            low, high = CONFIG_BOUNDS["llm_temperature"]
            validated["llm_temperature"] = _clamp(x, low, high)
        except (TypeError, ValueError):
            warnings.append("Invalid llm_temperature; using default")

    # llm_top_p
    v = settings.get("llm_top_p")
    if v is not None:
        try:
            x = float(v)
            low, high = CONFIG_BOUNDS["llm_top_p"]
            validated["llm_top_p"] = _clamp(x, low, high)
        except (TypeError, ValueError):
            warnings.append("Invalid llm_top_p; using default")

    # llm_n_gpu_layers
    v = settings.get("llm_n_gpu_layers")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["llm_n_gpu_layers"]
            validated["llm_n_gpu_layers"] = int(_clamp(n, low, high))
        except (TypeError, ValueError):
            warnings.append("Invalid llm_n_gpu_layers; using default")

    # llm_n_batch: prompt processing batch size (affects speed and memory)
    v = settings.get("llm_n_batch")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["llm_n_batch"]
            validated["llm_n_batch"] = int(_clamp(n, low, high))
            if n != validated["llm_n_batch"]:
                warnings.append(f"llm_n_batch clamped to {validated['llm_n_batch']}")
        except (TypeError, ValueError):
            warnings.append("Invalid llm_n_batch; using default")

    # llm_model_path: accept only string; path validation done in loading/paths
    v = settings.get("llm_model_path")
    if v is not None and isinstance(v, str) and v.strip():
        validated["llm_model_path"] = v.strip()

    # embedding_model: string, non-empty
    v = settings.get("embedding_model")
    if v is not None and isinstance(v, str) and v.strip():
        validated["embedding_model"] = v.strip()

    # encoding_model: tiktoken encoding name; ideally align with LLM tokenizer
    v = settings.get("encoding_model")
    if v is not None and isinstance(v, str) and v.strip():
        validated["encoding_model"] = v.strip()

    # word_fallback_chunk_ratio: when tiktoken unavailable, effective chunk size = chunk_size * this
    v = settings.get("word_fallback_chunk_ratio")
    if v is not None:
        try:
            x = float(v)
            low, high = CONFIG_BOUNDS["word_fallback_chunk_ratio"]
            validated["word_fallback_chunk_ratio"] = _clamp(x, low, high)
            if x != validated["word_fallback_chunk_ratio"]:
                warnings.append(
                    f"word_fallback_chunk_ratio clamped to {validated['word_fallback_chunk_ratio']}"
                )
        except (TypeError, ValueError):
            warnings.append("Invalid word_fallback_chunk_ratio; using default")

    # rag_context_order: "score" | "document_order"
    v = settings.get("rag_context_order")
    if v is not None and isinstance(v, str) and v.strip():
        v = v.strip().lower()
        if v in ("score", "document_order"):
            validated["rag_context_order"] = v
        else:
            warnings.append("Invalid rag_context_order; use 'score' or 'document_order'")

    # rag_rerank: bool
    v = settings.get("rag_rerank")
    if v is not None:
        validated["rag_rerank"] = bool(v) if not isinstance(v, bool) else v

    # rag_rerank_candidate_multiplier: int
    v = settings.get("rag_rerank_candidate_multiplier")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["rag_rerank_candidate_multiplier"]
            validated["rag_rerank_candidate_multiplier"] = int(_clamp(n, low, high))
        except (TypeError, ValueError):
            warnings.append("Invalid rag_rerank_candidate_multiplier; using default")

    # rag_max_chunks_per_doc: int >= 0; 0 = no cap; max chunks per document in final list
    v = settings.get("rag_max_chunks_per_doc")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["rag_max_chunks_per_doc"]
            validated["rag_max_chunks_per_doc"] = int(_clamp(n, low, high))
            if n != validated["rag_max_chunks_per_doc"]:
                warnings.append(
                    f"rag_max_chunks_per_doc clamped to {validated['rag_max_chunks_per_doc']}"
                )
        except (TypeError, ValueError):
            warnings.append("Invalid rag_max_chunks_per_doc; using default")

    # embedding_batch_size: int (chunks per batch when embedding documents)
    v = settings.get("embedding_batch_size")
    if v is not None:
        try:
            n = int(v)
            low, high = CONFIG_BOUNDS["embedding_batch_size"]
            validated["embedding_batch_size"] = int(_clamp(n, low, high))
            if n != validated["embedding_batch_size"]:
                warnings.append(
                    f"embedding_batch_size clamped to {validated['embedding_batch_size']}"
                )
        except (TypeError, ValueError):
            warnings.append("Invalid embedding_batch_size; using default")

    # embedding_show_progress: bool (show progress bar during document embedding)
    v = settings.get("embedding_show_progress")
    if v is not None:
        validated["embedding_show_progress"] = bool(v) if not isinstance(v, bool) else v

    # extractor_clean_text: bool (clean extracted text: dedup lines, merge hyphenation)
    v = settings.get("extractor_clean_text")
    if v is not None:
        validated["extractor_clean_text"] = bool(v) if not isinstance(v, bool) else v

    return validated, warnings


def get_default_settings() -> Dict[str, Any]:
    """Return default settings dict (values only; paths come from loading)."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "encoding_model": "cl100k_base",
        "word_fallback_chunk_ratio": 0.5,
        "top_k_retrieval": 5,
        "min_score_retrieval": 0.0,
        "llm_max_tokens": 256,
        "llm_temperature": 0.3,
        "llm_top_p": 0.9,
        "llm_n_gpu_layers": 0,
        "llm_n_batch": 512,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_batch_size": 32,
        "embedding_show_progress": True,
        "rag_context_order": "document_order",
        "rag_rerank": False,
        "rag_rerank_candidate_multiplier": 3,
        "rag_max_chunks_per_doc": 0,  # 0 = no cap; 2 or 3 to diversify across documents
        "extractor_clean_text": True,
    }
