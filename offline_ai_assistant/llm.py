"""
Local LLM module using llama-cpp-python for offline inference.

This module provides functionality to run local LLM models using GGUF format
with streaming support and configurable parameters.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List
import time
import threading
from dataclasses import dataclass

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None
    LlamaGrammar = None

from .config import Config

logger = logging.getLogger("OfflineAIAssistant.llm")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = None
    stream: bool = True
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


class LocalLLM:
    """Local LLM interface using llama-cpp-python."""
    
    def __init__(
        self,
        model_path: Path = None,
        n_ctx: int = None,
        n_gpu_layers: int = None,
        n_threads: int = None,
        verbose: bool = False
    ):
        """
        Initialize the local LLM.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context length
            n_gpu_layers: Number of layers to run on GPU
            n_threads: Number of CPU threads
            verbose: Enable verbose logging
        """
        self.model_path = model_path or Config.LLM_MODEL_PATH
        self.n_ctx = n_ctx or Config.LLM_CONTEXT_LENGTH
        self.n_gpu_layers = n_gpu_layers or Config.LLM_N_GPU_LAYERS
        self.n_threads = n_threads or Config.LLM_N_THREADS
        self.verbose = verbose
        
        self.model = None
        self.model_info = {}
        self._lock = threading.Lock()
        
        self._check_dependencies()
        self._load_model()
        
        logger.info(f"LocalLLM initialized: {self.model_path}")
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed. LLM inference will not work.")
    
    def _load_model(self) -> None:
        """Load the GGUF model."""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading LLM model: {self.model_path}")
        logger.info(f"Model size: {self.model_path.stat().st_size / (1024**3):.2f} GB")
        
        start_time = time.time()
        
        try:
            # Determine number of threads
            if self.n_threads is None:
                import os
                self.n_threads = max(1, os.cpu_count() // 2)
            
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose,
                use_mmap=True,
                use_mlock=False,  # Don't lock memory to allow swapping if needed
                n_batch=512,  # Batch size for prompt processing
                f16_kv=True,  # Use half precision for key/value cache
                logits_all=False,  # Don't compute logits for all tokens
                vocab_only=False,
                embedding=False
            )
            
            load_time = time.time() - start_time
            
            # Get model info
            self.model_info = {
                "model_path": str(self.model_path),
                "model_size_gb": self.model_path.stat().st_size / (1024**3),
                "context_length": self.n_ctx,
                "gpu_layers": self.n_gpu_layers,
                "cpu_threads": self.n_threads,
                "load_time": load_time,
                "vocab_size": self.model.n_vocab(),
                "embedding_dim": self.model.n_embd() if hasattr(self.model, 'n_embd') else None
            }
            
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Context length: {self.n_ctx}, GPU layers: {self.n_gpu_layers}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_path}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> Iterator[str]:
        """
        Generate text from a prompt with streaming.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Yields:
            Generated text tokens
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        config = config or GenerationConfig()
        
        logger.debug(f"Generating text for prompt: {prompt[:100]}...")
        
        with self._lock:
            try:
                start_time = time.time()
                
                # Generate with streaming
                response = self.model(
                    prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repeat_penalty,
                    stop=config.stop_sequences,
                    stream=config.stream,
                    echo=False  # Don't include prompt in output
                )
                
                if config.stream:
                    # Streaming response
                    full_response = ""
                    for chunk in response:
                        token = chunk['choices'][0]['text']
                        full_response += token
                        yield token
                    
                    generation_time = time.time() - start_time
                    tokens_generated = len(full_response.split())
                    
                    logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
                               f"({tokens_generated/generation_time:.1f} tokens/sec)")
                else:
                    # Non-streaming response
                    text = response['choices'][0]['text']
                    generation_time = time.time() - start_time
                    tokens_generated = len(text.split())
                    
                    logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f}s")
                    yield text
                    
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                raise RuntimeError(f"Text generation failed: {e}")
    
    def generate_complete(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> str:
        """
        Generate complete text from a prompt (non-streaming).
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Complete generated text
        """
        config = config or GenerationConfig()
        config.stream = False
        
        return next(self.generate(prompt, config))
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Number of tokens
        """
        if self.model is None:
            # Fallback to word count
            return len(text.split())
        
        try:
            tokens = self.model.tokenize(text.encode('utf-8'))
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text.split())
    
    def create_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_message: str = None
    ) -> str:
        """
        Create a chat prompt from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_message: Optional system message
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system message if provided
        if system_message:
            prompt_parts.append(f"System: {system_message}\n")
        
        # Add conversation messages
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                prompt_parts.append(f"Human: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}\n")
            elif role == 'system':
                prompt_parts.append(f"System: {content}\n")
        
        # Add assistant prompt
        prompt_parts.append("Assistant:")
        
        return "".join(prompt_parts)
    
    def create_rag_prompt(
        self,
        query: str,
        context_chunks: List[str],
        template: str = None
    ) -> str:
        """
        Create a RAG prompt with context and query.
        
        Args:
            query: User query
            context_chunks: List of relevant context chunks
            template: Prompt template
            
        Returns:
            Formatted RAG prompt
        """
        template = template or Config.PROMPT_TEMPLATES["default"]
        
        # Join context chunks
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Format template
        prompt = template.format(context=context, question=query)
        
        return prompt
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (faster approximation).
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 0.75 words
        word_count = len(text.split())
        return int(word_count * 1.33)
    
    def truncate_to_context(
        self,
        text: str,
        max_tokens: int = None,
        preserve_end: bool = True
    ) -> str:
        """
        Truncate text to fit within context window.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (defaults to 80% of context)
            preserve_end: Whether to preserve the end or beginning
            
        Returns:
            Truncated text
        """
        max_tokens = max_tokens or int(self.n_ctx * 0.8)
        
        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Simple word-based truncation
        words = text.split()
        target_words = int(max_tokens * 0.75)  # Conservative estimate
        
        if preserve_end:
            return " ".join(words[-target_words:])
        else:
            return " ".join(words[:target_words])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            **self.model_info
        }
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded
        """
        return self.model is not None
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                logger.info("Model unloaded")
    
    def reload_model(self) -> None:
        """Reload the model."""
        logger.info("Reloading model...")
        self.unload_model()
        self._load_model()


class LLMManager:
    """Manager for multiple LLM instances and configurations."""
    
    def __init__(self):
        """Initialize the LLM manager."""
        self.models = {}
        self.current_model = None
        self.default_config = GenerationConfig()
    
    def load_model(
        self,
        name: str,
        model_path: Path,
        **kwargs
    ) -> LocalLLM:
        """
        Load a model with a given name.
        
        Args:
            name: Model name
            model_path: Path to model file
            **kwargs: Additional model parameters
            
        Returns:
            LocalLLM instance
        """
        logger.info(f"Loading model '{name}' from {model_path}")
        
        model = LocalLLM(model_path=model_path, **kwargs)
        self.models[name] = model
        
        if self.current_model is None:
            self.current_model = name
        
        return model
    
    def get_model(self, name: str = None) -> Optional[LocalLLM]:
        """
        Get a model by name.
        
        Args:
            name: Model name (uses current if None)
            
        Returns:
            LocalLLM instance or None
        """
        if name is None:
            name = self.current_model
        
        return self.models.get(name)
    
    def set_current_model(self, name: str) -> bool:
        """
        Set the current active model.
        
        Args:
            name: Model name
            
        Returns:
            True if successful
        """
        if name in self.models:
            self.current_model = name
            return True
        return False
    
    def list_models(self) -> List[str]:
        """
        List available model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def unload_model(self, name: str) -> bool:
        """
        Unload a specific model.
        
        Args:
            name: Model name
            
        Returns:
            True if successful
        """
        if name in self.models:
            self.models[name].unload_model()
            del self.models[name]
            
            if self.current_model == name:
                self.current_model = next(iter(self.models.keys()), None)
            
            return True
        return False
    
    def unload_all(self) -> None:
        """Unload all models."""
        for model in self.models.values():
            model.unload_model()
        
        self.models.clear()
        self.current_model = None


def create_llm(
    model_path: Path = None,
    **kwargs
) -> LocalLLM:
    """
    Convenience function to create a LocalLLM instance.
    
    Args:
        model_path: Path to model file
        **kwargs: Additional parameters
        
    Returns:
        LocalLLM instance
    """
    return LocalLLM(model_path=model_path, **kwargs)
