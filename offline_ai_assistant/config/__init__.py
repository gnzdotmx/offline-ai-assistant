"""
Configuration with validation and secure path handling.

This package provides:
- Schema validation for config values (bounds, types)
- Path validation to keep file access under allowed directories
- Config loading/saving and logging setup
"""

from .loading import Config, setup_logging
from .paths import resolve_under, validate_path_under, SafePathResolver
from .schema import validate_settings, get_default_settings, CONFIG_BOUNDS

__all__ = [
    "Config",
    "setup_logging",
    "resolve_under",
    "validate_path_under",
    "SafePathResolver",
    "validate_settings",
    "get_default_settings",
    "CONFIG_BOUNDS",
]
