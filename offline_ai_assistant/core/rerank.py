"""
Re-ranking of retrieval results for better relevance.

Optional second stage after vector search: take a larger candidate set
(e.g. 2–3× top_k) and re-rank by a lightweight method (e.g. keyword overlap
or cross-encoder) before passing top_k to the prompt.
"""

import re
import logging
from typing import Any, Dict, List

from ..config import Config

logger = logging.getLogger("OfflineAIAssistant.rerank")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenize on non-alphanumeric; keep only tokens with at least one letter/digit."""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def _keyword_overlap_score(query_tokens: List[str], chunk_text: str) -> float:
    """
    Score chunk by overlap with query tokens (BM25-style simplicity: count matches).
    Normalize by query length so that more specific queries don't artificially dominate.
    """
    if not query_tokens:
        return 0.0
    chunk_tokens = set(_tokenize(chunk_text))
    if not chunk_tokens:
        return 0.0
    # Boost by frequency in chunk (simple TF), normalized by query length
    chunk_token_list = _tokenize(chunk_text)
    chunk_counts = {}
    for t in chunk_token_list:
        chunk_counts[t] = chunk_counts.get(t, 0) + 1
    score = 0.0
    for t in query_tokens:
        if t in chunk_counts:
            score += 1.0 + 0.5 * min(chunk_counts[t] - 1, 3)  # cap TF boost
    return score / max(len(query_tokens), 1)


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Re-rank candidate chunks by relevance to the query.

    If re-ranking is disabled (Config.RAG_RERANK is False), returns chunks
    unchanged up to top_k (no-op). Otherwise uses keyword-overlap scoring and
    returns up to top_k chunks in descending order of relevance.

    Args:
        query: User query string.
        chunks: List of chunk dicts from vector search (each must have "text" key).
        top_k: Number of chunks to return after re-ranking.

    Returns:
        List of up to top_k chunk dicts. Original keys preserved; optional
        "rerank_score" added when keyword re-ranker is used.
    """
    if not chunks or top_k <= 0:
        return chunks[:top_k] if top_k > 0 else []

    if not getattr(Config, "RAG_RERANK", False):
        return chunks[:top_k]

    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]

    scored: List[tuple] = []
    for i, c in enumerate(chunks):
        text = c.get("text") or ""
        score = _keyword_overlap_score(query_tokens, text)
        scored.append((score, i, c))

    scored.sort(key=lambda x: (-x[0], x[1]))

    result = []
    for score, _idx, c in scored[:top_k]:
        out = dict(c)
        out["rerank_score"] = score
        result.append(out)
    return result


def no_op_rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """No-op re-ranker: return chunks unchanged (up to top_k)."""
    return chunks[:top_k]
