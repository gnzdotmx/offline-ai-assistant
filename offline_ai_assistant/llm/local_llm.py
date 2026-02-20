"""
Local LLM using llama-cpp-python for offline inference.

Uses config for paths and defaults; GenerationConfig from core.models.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from ..config import Config
from ..core.models import GenerationConfig

logger = logging.getLogger("OfflineAIAssistant.llm")


class LocalLLM:
    """Local LLM interface using llama-cpp-python."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self.model_path = model_path or Config.LLM_MODEL_PATH
        self.n_ctx = n_ctx or Config.LLM_CONTEXT_LENGTH
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else Config.LLM_N_GPU_LAYERS
        self.n_batch = n_batch if n_batch is not None else Config.LLM_N_BATCH
        self.n_threads = n_threads or Config.LLM_N_THREADS
        self.verbose = verbose
        self.model = None
        self.model_info = {}
        self._lock = threading.Lock()
        self._check_dependencies()
        self._load_model()
        logger.info("LocalLLM initialized: %s", self.model_path)

    def _check_dependencies(self) -> None:
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed. LLM inference will not work.")

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.error("Model file not found: %s", self.model_path)
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        logger.info("Loading LLM model: %s", self.model_path)
        logger.info("Model size: %.2f GB", self.model_path.stat().st_size / (1024 ** 3))
        start_time = time.time()
        try:
            if self.n_threads is None:
                self.n_threads = max(1, (os.cpu_count() or 2) // 2)
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose,
                use_mmap=True,
                use_mlock=False,
                n_batch=self.n_batch,
                f16_kv=True,
                logits_all=False,
                vocab_only=False,
                embedding=False,
            )
            load_time = time.time() - start_time
            self.model_info = {
                "model_path": str(self.model_path),
                "model_size_gb": self.model_path.stat().st_size / (1024 ** 3),
                "context_length": self.n_ctx,
                "gpu_layers": self.n_gpu_layers,
                "cpu_threads": self.n_threads,
                "load_time": load_time,
                "vocab_size": self.model.n_vocab(),
                "embedding_dim": (lambda v: v() if callable(v) else v)(getattr(self.model, "n_embd", None)),
            }
            logger.info("Model loaded successfully in %.2fs", load_time)
        except Exception as e:
            logger.error("Error loading model %s: %s", self.model_path, e)
            raise RuntimeError(f"Failed to load model: {e}") from e

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        config = config or GenerationConfig()
        logger.debug("Generating text for prompt: %s...", prompt[:100])
        with self._lock:
            try:
                start_time = time.time()
                response = self.model(
                    prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repeat_penalty,
                    stop=config.stop_sequences,
                    stream=config.stream,
                    echo=False,
                )
                if config.stream:
                    full_response = ""
                    for chunk in response:
                        token = chunk["choices"][0]["text"]
                        full_response += token
                        yield token
                    logger.debug(
                        "Generated %s tokens in %.2fs",
                        len(full_response.split()),
                        time.time() - start_time,
                    )
                else:
                    text = response["choices"][0]["text"]
                    yield text
            except Exception as e:
                logger.error("Error generating text: %s", e)
                raise RuntimeError(f"Text generation failed: {e}") from e

    def generate_complete(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        config = config or GenerationConfig()
        config.stream = False
        return next(self.generate(prompt, config))

    def count_tokens(self, text: str) -> int:
        if self.model is None:
            return len(text.split())
        try:
            tokens = self.model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception as e:
            logger.warning("Error counting tokens: %s", e)
            return len(text.split())

    def create_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> str:
        prompt_parts = []
        if system_message:
            prompt_parts.append(f"System: {system_message}\n")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"Human: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            elif role == "system":
                prompt_parts.append(f"System: {content}\n")
        prompt_parts.append("Assistant:")
        return "".join(prompt_parts)

    def create_rag_prompt(
        self,
        query: str,
        context_chunks: List[str],
        template: Optional[str] = None,
    ) -> str:
        template = template or Config.PROMPT_TEMPLATES["default"]
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        return template.format(context=context, question=query)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using model tokenizer when available."""
        if self.model is not None:
            try:
                tokens = self.model.tokenize(text.encode("utf-8"))
                return len(tokens)
            except Exception as e:
                logger.warning("Token count via model failed, using word estimate: %s", e)
        return int(len(text.split()) * 1.33)

    def _decode_tokens_to_str(self, raw: bytes) -> str:
        """Decode tokenizer output to string, handling partial/invalid UTF-8."""
        return raw.decode("utf-8", errors="replace")

    def truncate_to_context(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_end: bool = True,
    ) -> str:
        max_tokens = max_tokens or int(self.n_ctx * 0.8)
        if self.model is not None:
            try:
                tokens = self.model.tokenize(text.encode("utf-8"))
                if len(tokens) <= max_tokens:
                    return text
                subset = tokens[-max_tokens:] if preserve_end else tokens[:max_tokens]
                raw = self.model.detokenize(subset)
                return self._decode_tokens_to_str(raw)
            except Exception as e:
                logger.warning("Token-based truncation failed, using word-based: %s", e)
        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        words = text.split()
        target_words = int(max_tokens * 0.75)
        if preserve_end:
            return " ".join(words[-target_words:])
        return " ".join(words[:target_words])

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "not_loaded"}
        return {"status": "loaded", **self.model_info}

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload_model(self) -> None:
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                logger.info("Model unloaded")

    def reload_model(self) -> None:
        logger.info("Reloading model...")
        self.unload_model()
        self._load_model()


class LLMManager:
    """Manager for multiple LLM instances."""

    def __init__(self) -> None:
        self.models: Dict[str, LocalLLM] = {}
        self.current_model: Optional[str] = None
        self.default_config = GenerationConfig()

    def load_model(self, name: str, model_path: Path, **kwargs: Any) -> LocalLLM:
        logger.info("Loading model '%s' from %s", name, model_path)
        model = LocalLLM(model_path=model_path, **kwargs)
        self.models[name] = model
        if self.current_model is None:
            self.current_model = name
        return model

    def get_model(self, name: Optional[str] = None) -> Optional[LocalLLM]:
        if name is None:
            name = self.current_model
        return self.models.get(name) if name else None

    def set_current_model(self, name: str) -> bool:
        if name in self.models:
            self.current_model = name
            return True
        return False

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def unload_model(self, name: str) -> bool:
        if name in self.models:
            self.models[name].unload_model()
            del self.models[name]
            if self.current_model == name:
                self.current_model = next(iter(self.models.keys()), None)
            return True
        return False

    def unload_all(self) -> None:
        for model in self.models.values():
            model.unload_model()
        self.models.clear()
        self.current_model = None


def create_llm(model_path: Optional[Path] = None, **kwargs: Any) -> LocalLLM:
    return LocalLLM(model_path=model_path, **kwargs)
