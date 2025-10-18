"""
Configuration module for the Offline AI Assistant.

This module contains all configuration settings, paths, and constants
used throughout the application.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class Config:
    """Configuration class for the Offline AI Assistant."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    APP_DIR = Path(__file__).parent
    
    # User data directory (cross-platform)
    USER_DATA_DIR = Path.home() / ".config" / "ai-offline-assistant"
    
    # Configuration file
    CONFIG_FILE = USER_DATA_DIR / "config.json"
    
    # Data directories
    DB_DIR = USER_DATA_DIR / "db"
    DOCS_DIR = USER_DATA_DIR / "docs"
    MODELS_DIR = USER_DATA_DIR / "models"
    LOGS_DIR = USER_DATA_DIR / "logs"
    
    # Database files
    FAISS_INDEX_PATH = DB_DIR / "faiss_index"
    SQLITE_DB_PATH = DB_DIR / "metadata.db"
    
    # Model paths
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_CACHE = MODELS_DIR / "sentence-transformers"
    LLM_MODEL_PATH = MODELS_DIR / "llama-2-7b-chat.Q4_K_M.gguf"
    
    # Chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    ENCODING_MODEL = "cl100k_base"  # GPT-4 tokenizer
    
    # RAG settings
    TOP_K_RETRIEVAL = 5
    MAX_CONTEXT_LENGTH = 2048
    
    # LLM settings
    LLM_MAX_TOKENS = 256  # Reduced for faster responses
    LLM_TEMPERATURE = 0.3  # Lower temperature for more focused responses
    LLM_TOP_P = 0.9
    LLM_CONTEXT_LENGTH = 4096
    LLM_N_GPU_LAYERS = 0  # CPU-only by default
    LLM_N_THREADS = None  # Auto-detect
    
    # UI settings
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    WINDOW_MIN_WIDTH = 800
    WINDOW_MIN_HEIGHT = 600
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
    
    # Logging settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "app.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Prompt templates (optimized for concise responses)
    PROMPT_TEMPLATES = {
        "default": """Answer concisely based on the context. Be direct and specific.

Context:
{context}

Question: {question}

Answer (keep it brief):""",
        
        "summary": """Provide a brief summary of the key points:

{context}

Summary:""",
        
        "project_plan": """Create a concise project plan with main steps:

Context:
{context}

Question: {question}

Plan:""",
        
        "executive_briefing": """Brief executive summary:

Context:
{context}

Topic: {question}

Key Points:"""
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
            "top_k_retrieval": cls.TOP_K_RETRIEVAL,
            "llm_max_tokens": cls.LLM_MAX_TOKENS,
            "llm_temperature": cls.LLM_TEMPERATURE,
            "llm_top_p": cls.LLM_TOP_P,
            "llm_n_gpu_layers": cls.LLM_N_GPU_LAYERS,
            "llm_model_path": str(cls.LLM_MODEL_PATH),
            "embedding_model": cls.EMBEDDING_MODEL_NAME
        }
    
    @classmethod
    def update_settings(cls, settings: Dict[str, Any], save: bool = True) -> None:
        """
        Update configuration settings.
        
        Args:
            settings: Dictionary of settings to update
            save: Whether to save settings to disk (default: True)
        """
        if "chunk_size" in settings:
            cls.CHUNK_SIZE = int(settings["chunk_size"])
        if "chunk_overlap" in settings:
            cls.CHUNK_OVERLAP = int(settings["chunk_overlap"])
        if "top_k_retrieval" in settings:
            cls.TOP_K_RETRIEVAL = int(settings["top_k_retrieval"])
        if "llm_max_tokens" in settings:
            cls.LLM_MAX_TOKENS = int(settings["llm_max_tokens"])
        if "llm_temperature" in settings:
            cls.LLM_TEMPERATURE = float(settings["llm_temperature"])
        if "llm_top_p" in settings:
            cls.LLM_TOP_P = float(settings["llm_top_p"])
        if "llm_n_gpu_layers" in settings:
            cls.LLM_N_GPU_LAYERS = int(settings["llm_n_gpu_layers"])
        if "llm_model_path" in settings:
            cls.LLM_MODEL_PATH = Path(settings["llm_model_path"])
        
        # Save to disk if requested
        if save:
            cls.save_config()
    
    @classmethod
    def load_config(cls) -> None:
        """Load configuration from disk if it exists."""
        if not cls.CONFIG_FILE.exists():
            return
        
        try:
            with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved_settings = json.load(f)
            
            # Update settings without saving (to avoid recursion)
            cls.update_settings(saved_settings, save=False)
            
            logger = logging.getLogger("OfflineAIAssistant.config")
            logger.info(f"Configuration loaded from {cls.CONFIG_FILE}")
            
        except Exception as e:
            logger = logging.getLogger("OfflineAIAssistant.config")
            logger.error(f"Failed to load configuration: {e}")
    
    @classmethod
    def save_config(cls) -> None:
        """Save current configuration to disk."""
        try:
            # Ensure directory exists
            cls.USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Get current settings
            settings = cls.get_settings_dict()
            
            # Save to file
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            
            logger = logging.getLogger("OfflineAIAssistant.config")
            logger.info(f"Configuration saved to {cls.CONFIG_FILE}")
            
        except Exception as e:
            logger = logging.getLogger("OfflineAIAssistant.config")
            logger.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset configuration to default values."""
        # Chunking settings
        cls.CHUNK_SIZE = 512
        cls.CHUNK_OVERLAP = 50
        
        # RAG settings
        cls.TOP_K_RETRIEVAL = 5
        
        # LLM settings
        cls.LLM_MAX_TOKENS = 256
        cls.LLM_TEMPERATURE = 0.3
        cls.LLM_TOP_P = 0.9
        cls.LLM_N_GPU_LAYERS = 0
        cls.LLM_MODEL_PATH = cls.MODELS_DIR / "llama-2-7b-chat.Q4_K_M.gguf"
        
        # Save the defaults
        cls.save_config()
        
        logger = logging.getLogger("OfflineAIAssistant.config")
        logger.info("Configuration reset to defaults")


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    Config.ensure_directories()
    
    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # Create file handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=Config.MAX_LOG_SIZE,
        backupCount=Config.LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(Config.LOG_LEVEL)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(Config.LOG_LEVEL)
    
    # Create logger
    logger = logging.getLogger("OfflineAIAssistant")
    logger.setLevel(Config.LOG_LEVEL)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
