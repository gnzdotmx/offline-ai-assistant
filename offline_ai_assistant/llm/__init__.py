"""
Local LLM layer using llama-cpp-python.

GenerationConfig is defined in core.models; this package provides LocalLLM and LLMManager.
"""

from .local_llm import LocalLLM, LLMManager, create_llm

__all__ = ["LocalLLM", "LLMManager", "create_llm"]
