"""
Secure path resolution and validation.

Ensures all file access stays under allowed base directories to prevent
path traversal and access to sensitive locations. Use these helpers
whenever resolving user-provided or config paths.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("OfflineAIAssistant.config.paths")


def resolve_under(candidate: Path, base: Path) -> Optional[Path]:
    """
    Resolve candidate path and return it only if it is under base (or equal).
    Returns None if resolution fails or path escapes base.
    """
    try:
        base_resolved = base.resolve()
        cand_resolved = Path(candidate).resolve()
        # For files, ensure parent is under base; for dirs, ensure self is under base
        try:
            cand_resolved.relative_to(base_resolved)
        except ValueError:
            return None
        return cand_resolved
    except (OSError, RuntimeError) as e:
        logger.debug("Path resolution failed: %s", e)
        return None


def validate_path_under(candidate: Path, base: Path, must_exist: bool = False) -> tuple:
    """
    Validate that candidate is under base. Returns (is_valid: bool, resolved_path or error_message: str).
    If must_exist is True, path must exist.
    """
    resolved = resolve_under(candidate, base)
    if resolved is None:
        return False, f"Path is not under allowed base: {base}"
    if must_exist and not resolved.exists():
        return False, f"Path does not exist: {resolved}"
    return True, str(resolved)


class SafePathResolver:
    """
    Resolves paths relative to configured base directories.
    Use for docs dir, models dir, and LLM model path validation.
    """

    def __init__(self, user_data_dir: Path, models_dir: Path, docs_dir: Path):
        self.user_data_dir = user_data_dir.resolve()
        self.models_dir = models_dir.resolve()
        self.docs_dir = docs_dir.resolve()

    def resolve_docs_path(self, path: Path) -> Optional[Path]:
        """Return path if it is under docs_dir, else None."""
        return resolve_under(path, self.docs_dir)

    def resolve_models_path(self, path: Path) -> Optional[Path]:
        """Return path if it is under models_dir (or user_data_dir for flexibility), else None."""
        return resolve_under(path, self.models_dir) or resolve_under(path, self.user_data_dir)

    def resolve_llm_model_path(self, path: Path) -> Optional[Path]:
        """
        Validate LLM model path: must be under models_dir (or user_data_dir)
        and typically a file. Returns resolved path or None.
        """
        return self.resolve_models_path(path)

    def safe_join_docs(self, *parts: str) -> Path:
        """Join parts under docs_dir and return path (caller should not use .. in parts)."""
        return self.docs_dir.joinpath(*parts)

    def safe_join_models(self, *parts: str) -> Path:
        """Join parts under models_dir."""
        return self.models_dir.joinpath(*parts)
