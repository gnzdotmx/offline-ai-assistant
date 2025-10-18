"""
Model manager module for downloading and managing LLM models.

This module provides functionality to download, verify, and manage GGUF models
from Hugging Face and other sources.
"""

import logging
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import json

from .config import Config

logger = logging.getLogger("OfflineAIAssistant.model_manager")


@dataclass
class ModelInfo:
    """Information about an available model."""
    
    id: str
    name: str
    description: str
    size_gb: float
    url: str
    filename: str
    sha256: str = ""
    quantization: str = "Q4_K_M"
    context_length: int = 4096


class ModelManager:
    """Manager for downloading and managing LLM models."""
    
    # Available models catalog
    AVAILABLE_MODELS = {
        "llama-2-7b-chat-q4": ModelInfo(
            id="llama-2-7b-chat-q4",
            name="Llama 2 7B Chat (Q4_K_M)",
            description="Small, fast model good for general use",
            size_gb=3.8,
            url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            filename="llama-2-7b-chat.Q4_K_M.gguf",
            quantization="Q4_K_M",
            context_length=4096
        ),
        "mistral-7b-instruct-q4": ModelInfo(
            id="mistral-7b-instruct-q4",
            name="Mistral 7B Instruct (Q4_K_M)",
            description="Efficient 7B model with good performance",
            size_gb=4.1,
            url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            quantization="Q4_K_M",
            context_length=8192
        ),
        "codellama-7b-instruct-q4": ModelInfo(
            id="codellama-7b-instruct-q4",
            name="Code Llama 7B Instruct (Q4_K_M)",
            description="Optimized for code and technical content",
            size_gb=3.8,
            url="https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf",
            filename="codellama-7b-instruct.Q4_K_M.gguf",
            quantization="Q4_K_M",
            context_length=16384
        ),
        "phi-2-q4": ModelInfo(
            id="phi-2-q4",
            name="Phi-2 (Q4_K_M)",
            description="Tiny but capable model (2.7B params)",
            size_gb=1.6,
            url="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
            filename="phi-2.Q4_K_M.gguf",
            quantization="Q4_K_M",
            context_length=2048
        ),
    }
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store models (defaults to Config.MODELS_DIR)
        """
        self.models_dir = models_dir or Config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model metadata file
        self.metadata_file = self.models_dir / "models_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"ModelManager initialized: models_dir={self.models_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load model metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
    
    def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available models for download.
        
        Returns:
            List of ModelInfo objects
        """
        return list(self.AVAILABLE_MODELS.values())
    
    def get_installed_models(self) -> List[Dict]:
        """
        Get list of installed models.
        
        Returns:
            List of dictionaries with model information
        """
        installed = []
        
        for model_file in self.models_dir.glob("*.gguf"):
            model_id = None
            model_info = None
            
            # Try to match with known models
            for mid, info in self.AVAILABLE_MODELS.items():
                if info.filename == model_file.name:
                    model_id = mid
                    model_info = info
                    break
            
            # Get metadata if available
            metadata = self.metadata.get(model_file.name, {})
            
            installed.append({
                "id": model_id,
                "filename": model_file.name,
                "path": str(model_file),
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "name": model_info.name if model_info else model_file.stem,
                "description": model_info.description if model_info else "Custom model",
                "downloaded_at": metadata.get("downloaded_at", "Unknown")
            })
        
        return installed
    
    def is_model_installed(self, model_id: str) -> bool:
        """
        Check if a model is installed.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model is installed
        """
        if model_id not in self.AVAILABLE_MODELS:
            return False
        
        model_info = self.AVAILABLE_MODELS[model_id]
        model_path = self.models_dir / model_info.filename
        
        return model_path.exists()
    
    def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Download a model from the catalog.
        
        Args:
            model_id: Model identifier
            progress_callback: Optional callback(downloaded_bytes, total_bytes)
            
        Returns:
            True if successful
        """
        if model_id not in self.AVAILABLE_MODELS:
            logger.error(f"Unknown model: {model_id}")
            return False
        
        model_info = self.AVAILABLE_MODELS[model_id]
        destination = self.models_dir / model_info.filename
        
        # Check if already exists
        if destination.exists():
            logger.warning(f"Model already exists: {destination}")
            return True
        
        logger.info(f"Downloading model: {model_info.name}")
        logger.info(f"From: {model_info.url}")
        logger.info(f"To: {destination}")
        
        try:
            # Download with progress
            response = requests.get(model_info.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Create temporary file
            temp_file = destination.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)
            
            # Verify file size
            if total_size > 0 and abs(temp_file.stat().st_size - total_size) > 1024 * 1024:
                logger.error(f"Downloaded file size mismatch: expected {total_size}, got {temp_file.stat().st_size}")
                temp_file.unlink()
                return False
            
            # Rename to final destination
            temp_file.rename(destination)
            
            # Update metadata
            from datetime import datetime
            self.metadata[model_info.filename] = {
                "model_id": model_id,
                "downloaded_at": datetime.now().isoformat(),
                "size_bytes": destination.stat().st_size,
                "url": model_info.url
            }
            self._save_metadata()
            
            logger.info(f"Model downloaded successfully: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            # Clean up partial download
            if destination.exists():
                destination.unlink()
            temp_file = destination.with_suffix('.tmp')
            if temp_file.exists():
                temp_file.unlink()
            return False
    
    def delete_model(self, filename: str) -> bool:
        """
        Delete a model file.
        
        Args:
            filename: Model filename to delete
            
        Returns:
            True if successful
        """
        model_path = self.models_dir / filename
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            model_path.unlink()
            
            # Remove from metadata
            if filename in self.metadata:
                del self.metadata[filename]
                self._save_metadata()
            
            logger.info(f"Model deleted: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Get the path to an installed model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model file or None if not installed
        """
        if model_id not in self.AVAILABLE_MODELS:
            return None
        
        model_info = self.AVAILABLE_MODELS[model_id]
        model_path = self.models_dir / model_info.filename
        
        return model_path if model_path.exists() else None
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash: {e}")
            return ""
    
    def verify_model(self, model_id: str) -> bool:
        """
        Verify model file integrity (if hash is available).
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if verification passes or hash not available
        """
        if model_id not in self.AVAILABLE_MODELS:
            return False
        
        model_info = self.AVAILABLE_MODELS[model_id]
        
        if not model_info.sha256:
            logger.info("No hash available for verification")
            return True
        
        model_path = self.get_model_path(model_id)
        if not model_path:
            return False
        
        logger.info(f"Verifying model: {model_info.name}")
        file_hash = self.calculate_file_hash(model_path)
        
        if file_hash == model_info.sha256:
            logger.info("Model verification passed")
            return True
        else:
            logger.warning(f"Model verification failed: hash mismatch")
            return False


def create_model_manager() -> ModelManager:
    """
    Convenience function to create a model manager.
    
    Returns:
        ModelManager instance
    """
    return ModelManager()

