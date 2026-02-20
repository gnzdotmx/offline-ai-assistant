"""
Configuration loading, saving, and logging setup.

Uses schema validation and path validation for security.
Config remains a class with class attributes for backward compatibility.

Environment:
  OFFLINE_AI_MODELS_DIR  Optional. Custom folder for LLM and embedding models.
                         If set, GGUF models and sentence-transformers cache use this dir.
                         Default: ~/.config/ai-offline-assistant/models
  OFFLINE_AI_DISABLE_EMBEDDING_CACHE  Optional. Set to 1, true, or yes to disable
                         embedding cache (chunk and query embeddings recomputed every time).
                         Default: cache enabled.
  OFFLINE_AI_EMBEDDING_BATCH_SIZE  Optional. Chunk batch size for document embedding (1–512).
                         Lower values reduce memory use; higher can speed up large docs.
                         Default: 32 (or value from config.json).
  OFFLINE_AI_EMBEDDING_SHOW_PROGRESS  Optional. Set to 0, false, or no to disable
                         progress bar during document embedding (e.g. headless/quiet mode).
                         Default: true (or value from config.json).
  OFFLINE_AI_LLM_N_BATCH  Optional. Prompt processing batch size for the LLM (64–2048).
                         Higher values can speed up long prompts on capable hardware but use more memory.
                         Default: 512 (or value from config.json).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

from .schema import validate_settings, get_default_settings, CONFIG_BOUNDS
from .paths import SafePathResolver, resolve_under

# Base paths (not user-configurable)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_APP_DIR = Path(__file__).resolve().parent.parent
_USER_DATA_DIR = Path.home() / ".config" / "ai-offline-assistant"

# Models dir: use OFFLINE_AI_MODELS_DIR if set (e.g. shared models with other projects), else default
_env_models = os.environ.get("OFFLINE_AI_MODELS_DIR", "").strip()
_MODELS_DIR = Path(_env_models).resolve() if _env_models else (_USER_DATA_DIR / "models")

# Embedding cache: disabled if OFFLINE_AI_DISABLE_EMBEDDING_CACHE is 1, true, or yes
_env_disable_emb_cache = os.environ.get("OFFLINE_AI_DISABLE_EMBEDDING_CACHE", "").strip().lower()
_DISABLE_EMBEDDING_CACHE = _env_disable_emb_cache in ("1", "true", "yes")

# Embedding batch size: env override (validated when applied in load_config)
_env_embedding_batch_size = os.environ.get("OFFLINE_AI_EMBEDDING_BATCH_SIZE", "").strip()

# Embedding show progress: 0/false/no disables progress bar (e.g. headless/quiet)
_env_embedding_show_progress = os.environ.get("OFFLINE_AI_EMBEDDING_SHOW_PROGRESS", "").strip().lower()
_EMBEDDING_SHOW_PROGRESS_ENV_FALSE = _env_embedding_show_progress in ("0", "false", "no")

# LLM n_batch: prompt processing batch size (env override applied in load_config)
_env_llm_n_batch = os.environ.get("OFFLINE_AI_LLM_N_BATCH", "").strip()

# Will be set after first load
_path_resolver: SafePathResolver = None


def _get_path_resolver() -> SafePathResolver:
    global _path_resolver
    if _path_resolver is None:
        _path_resolver = SafePathResolver(
            user_data_dir=_USER_DATA_DIR,
            models_dir=Config.MODELS_DIR,
            docs_dir=_USER_DATA_DIR / "docs",
        )
    return _path_resolver


def _apply_embedding_env_overrides() -> None:
    """Apply env overrides for embedding batch size and show_progress (no save)."""
    if _env_embedding_batch_size:
        try:
            n = int(_env_embedding_batch_size)
            low, high = CONFIG_BOUNDS["embedding_batch_size"]
            Config.EMBEDDING_BATCH_SIZE = max(low, min(high, n))
        except ValueError:
            pass
    if _EMBEDDING_SHOW_PROGRESS_ENV_FALSE:
        Config.EMBEDDING_SHOW_PROGRESS = False
    if _env_llm_n_batch:
        try:
            n = int(_env_llm_n_batch)
            low, high = CONFIG_BOUNDS["llm_n_batch"]
            Config.LLM_N_BATCH = max(low, min(high, n))
        except ValueError:
            pass


class Config:
    """Configuration class for the Offline AI Assistant (validated, path-safe)."""

    # Base paths (read-only)
    BASE_DIR = _BASE_DIR
    APP_DIR = _APP_DIR
    USER_DATA_DIR = _USER_DATA_DIR

    # Data directories (docs/db/logs under USER_DATA_DIR; models can be overridden by env)
    DB_DIR = _USER_DATA_DIR / "db"
    DOCS_DIR = _USER_DATA_DIR / "docs"
    MODELS_DIR = _MODELS_DIR
    LOGS_DIR = _USER_DATA_DIR / "logs"

    CONFIG_FILE = _USER_DATA_DIR / "config.json"
    FAISS_INDEX_PATH = DB_DIR / "faiss_index"
    SQLITE_DB_PATH = DB_DIR / "metadata.db"

    # Env-only: disable embedding cache (no persistence in config.json)
    DISABLE_EMBEDDING_CACHE = _DISABLE_EMBEDDING_CACHE

    # Defaults (overridden by validated load)
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_CACHE = MODELS_DIR / "sentence-transformers"
    LLM_MODEL_PATH = MODELS_DIR / "llama-2-7b-chat.Q4_K_M.gguf"

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    ENCODING_MODEL = "cl100k_base"
    WORD_FALLBACK_CHUNK_RATIO = 0.5  # when tiktoken unavailable, use chunk_size * this for word-based chunking
    TOP_K_RETRIEVAL = 5
    MIN_SCORE_RETRIEVAL = 0.0  # minimum similarity for retrieved chunks; -1 = no filtering
    MAX_CONTEXT_LENGTH = 2048
    RAG_CONTEXT_ORDER = "document_order"  # "score" | "document_order"
    RAG_RERANK = False  # Optional re-rank after vector search (keyword overlap, no extra deps)
    RAG_RERANK_CANDIDATE_MULTIPLIER = 3  # Retrieve top_k * this many, then re-rank to top_k
    RAG_MAX_CHUNKS_PER_DOC = 0  # Max chunks per document in final list; 0 = no cap (preserve current behavior)

    EMBEDDING_BATCH_SIZE = 32  # Chunks per batch when embedding documents (1–512)
    EMBEDDING_SHOW_PROGRESS = True  # Show progress bar during document embedding; set False for headless/quiet

    LLM_MAX_TOKENS = 256
    LLM_TEMPERATURE = 0.3
    LLM_TOP_P = 0.9
    LLM_CONTEXT_LENGTH = 4096
    LLM_N_GPU_LAYERS = 0
    LLM_N_BATCH = 512  # Prompt processing batch size; higher can speed long prompts, uses more memory
    LLM_N_THREADS = None

    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    WINDOW_MIN_WIDTH = 800
    WINDOW_MIN_HEIGHT = 600

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
    EXTRACTOR_CLEAN_TEXT = True  # Clean extracted text (dedup lines, merge hyphenation); set False for raw

    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "app.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    PROMPT_TEMPLATES = {
        "default": """Answer concisely based on the context. Be direct and specific.
When possible, cite the context by number, e.g. [1], [2].

Context:
{context}

Question: {question}

Answer (keep it brief):""",
        "summary": """Provide a brief summary of the key points:

{context}

Summary:""",
        "project_plan": """Create a concise project plan with main steps.
When possible, cite the context by number, e.g. [1], [2].

Context:
{context}

Question: {question}

Plan:""",
        "executive_briefing": """Brief executive summary.
When possible, cite the context by number, e.g. [1], [2].

Context:
{context}

Topic: {question}

Key Points:""",
    }

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        for directory in [cls.DB_DIR, cls.DOCS_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_settings_dict(cls) -> Dict[str, Any]:
        """Get configuration as a dictionary for UI display."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "encoding_model": cls.ENCODING_MODEL,
            "word_fallback_chunk_ratio": cls.WORD_FALLBACK_CHUNK_RATIO,
            "top_k_retrieval": cls.TOP_K_RETRIEVAL,
            "min_score_retrieval": cls.MIN_SCORE_RETRIEVAL,
            "llm_max_tokens": cls.LLM_MAX_TOKENS,
            "llm_temperature": cls.LLM_TEMPERATURE,
            "llm_top_p": cls.LLM_TOP_P,
            "llm_n_gpu_layers": cls.LLM_N_GPU_LAYERS,
            "llm_n_batch": cls.LLM_N_BATCH,
            "llm_model_path": str(cls.LLM_MODEL_PATH),
            "embedding_model": cls.EMBEDDING_MODEL_NAME,
            "rag_context_order": cls.RAG_CONTEXT_ORDER,
            "rag_rerank": cls.RAG_RERANK,
            "rag_rerank_candidate_multiplier": cls.RAG_RERANK_CANDIDATE_MULTIPLIER,
            "rag_max_chunks_per_doc": cls.RAG_MAX_CHUNKS_PER_DOC,
            "embedding_batch_size": cls.EMBEDDING_BATCH_SIZE,
            "embedding_show_progress": cls.EMBEDDING_SHOW_PROGRESS,
            "extractor_clean_text": cls.EXTRACTOR_CLEAN_TEXT,
        }

    @classmethod
    def update_settings(cls, settings: Dict[str, Any], save: bool = True) -> None:
        """Update configuration with validated values."""
        validated, warnings = validate_settings(settings)
        log = logging.getLogger("OfflineAIAssistant.config")
        for w in warnings:
            log.warning(w)

        defaults = get_default_settings()
        if "chunk_size" in validated:
            cls.CHUNK_SIZE = validated["chunk_size"]
        if "chunk_overlap" in validated:
            cls.CHUNK_OVERLAP = validated["chunk_overlap"]
        if "encoding_model" in validated:
            cls.ENCODING_MODEL = validated["encoding_model"]
        if "word_fallback_chunk_ratio" in validated:
            cls.WORD_FALLBACK_CHUNK_RATIO = float(validated["word_fallback_chunk_ratio"])
        if "top_k_retrieval" in validated:
            cls.TOP_K_RETRIEVAL = validated["top_k_retrieval"]
        if "min_score_retrieval" in validated:
            cls.MIN_SCORE_RETRIEVAL = float(validated["min_score_retrieval"])
        if "llm_max_tokens" in validated:
            cls.LLM_MAX_TOKENS = validated["llm_max_tokens"]
        if "llm_temperature" in validated:
            cls.LLM_TEMPERATURE = validated["llm_temperature"]
        if "llm_top_p" in validated:
            cls.LLM_TOP_P = validated["llm_top_p"]
        if "llm_n_gpu_layers" in validated:
            cls.LLM_N_GPU_LAYERS = validated["llm_n_gpu_layers"]
        if "llm_n_batch" in validated:
            cls.LLM_N_BATCH = validated["llm_n_batch"]
        if "embedding_model" in validated:
            cls.EMBEDDING_MODEL_NAME = validated["embedding_model"]
        if "rag_context_order" in validated:
            cls.RAG_CONTEXT_ORDER = validated["rag_context_order"]
        if "rag_rerank" in validated:
            cls.RAG_RERANK = validated["rag_rerank"]
        if "rag_rerank_candidate_multiplier" in validated:
            cls.RAG_RERANK_CANDIDATE_MULTIPLIER = validated["rag_rerank_candidate_multiplier"]
        if "rag_max_chunks_per_doc" in validated:
            cls.RAG_MAX_CHUNKS_PER_DOC = validated["rag_max_chunks_per_doc"]
        if "embedding_batch_size" in validated:
            cls.EMBEDDING_BATCH_SIZE = validated["embedding_batch_size"]
        if "embedding_show_progress" in validated:
            cls.EMBEDDING_SHOW_PROGRESS = validated["embedding_show_progress"]
        if "extractor_clean_text" in validated:
            cls.EXTRACTOR_CLEAN_TEXT = validated["extractor_clean_text"]

        if "llm_model_path" in validated:
            raw = validated["llm_model_path"]
            resolver = _get_path_resolver()
            resolved = resolve_under(Path(raw), cls.MODELS_DIR) or resolve_under(
                Path(raw), cls.USER_DATA_DIR
            )
            if resolved is not None:
                cls.LLM_MODEL_PATH = resolved
            else:
                # Allow path as-is but log; UI may show path outside sandbox
                cls.LLM_MODEL_PATH = Path(raw)
                log.warning("llm_model_path is outside allowed dirs: %s", raw)

        if save:
            cls.save_config()

    @classmethod
    def load_config(cls) -> None:
        """Load configuration from disk with validation."""
        if not cls.CONFIG_FILE.exists():
            _apply_embedding_env_overrides()
            return
        try:
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                saved_settings = json.load(f)
            if isinstance(saved_settings, dict):
                cls.update_settings(saved_settings, save=False)
            _apply_embedding_env_overrides()
            log = logging.getLogger("OfflineAIAssistant.config")
            log.info("Configuration loaded from %s", cls.CONFIG_FILE)
        except json.JSONDecodeError as e:
            log = logging.getLogger("OfflineAIAssistant.config")
            log.error("Invalid config file (JSON): %s", e)
        except OSError as e:
            log = logging.getLogger("OfflineAIAssistant.config")
            log.error("Failed to load configuration: %s", e)

    @classmethod
    def save_config(cls) -> None:
        """Save current configuration to disk."""
        try:
            cls.USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            settings = cls.get_settings_dict()
            with open(cls.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            log = logging.getLogger("OfflineAIAssistant.config")
            log.info("Configuration saved to %s", cls.CONFIG_FILE)
        except OSError as e:
            log = logging.getLogger("OfflineAIAssistant.config")
            log.error("Failed to save configuration: %s", e)

    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset configuration to default values."""
        defaults = get_default_settings()
        cls.CHUNK_SIZE = defaults["chunk_size"]
        cls.CHUNK_OVERLAP = defaults["chunk_overlap"]
        cls.ENCODING_MODEL = defaults.get("encoding_model", "cl100k_base")
        cls.WORD_FALLBACK_CHUNK_RATIO = defaults.get("word_fallback_chunk_ratio", 0.5)
        cls.TOP_K_RETRIEVAL = defaults["top_k_retrieval"]
        cls.MIN_SCORE_RETRIEVAL = defaults.get("min_score_retrieval", 0.0)
        cls.LLM_MAX_TOKENS = defaults["llm_max_tokens"]
        cls.LLM_TEMPERATURE = defaults["llm_temperature"]
        cls.LLM_TOP_P = defaults["llm_top_p"]
        cls.LLM_N_GPU_LAYERS = defaults["llm_n_gpu_layers"]
        cls.LLM_N_BATCH = defaults.get("llm_n_batch", 512)
        cls.EMBEDDING_MODEL_NAME = defaults["embedding_model"]
        cls.LLM_MODEL_PATH = cls.MODELS_DIR / "llama-2-7b-chat.Q4_K_M.gguf"
        cls.RAG_CONTEXT_ORDER = defaults.get("rag_context_order", "document_order")
        cls.RAG_RERANK = defaults.get("rag_rerank", False)
        cls.RAG_RERANK_CANDIDATE_MULTIPLIER = defaults.get("rag_rerank_candidate_multiplier", 3)
        cls.RAG_MAX_CHUNKS_PER_DOC = defaults.get("rag_max_chunks_per_doc", 0)
        cls.EMBEDDING_BATCH_SIZE = defaults.get("embedding_batch_size", 32)
        cls.EMBEDDING_SHOW_PROGRESS = defaults.get("embedding_show_progress", True)
        cls.EXTRACTOR_CLEAN_TEXT = defaults.get("extractor_clean_text", True)
        cls.save_config()
        log = logging.getLogger("OfflineAIAssistant.config")
        log.info("Configuration reset to defaults")


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    Config.ensure_directories()
    formatter = logging.Formatter(Config.LOG_FORMAT)
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=Config.MAX_LOG_SIZE,
        backupCount=Config.LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(Config.LOG_LEVEL)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(Config.LOG_LEVEL)
    logger = logging.getLogger("OfflineAIAssistant")
    logger.setLevel(Config.LOG_LEVEL)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
