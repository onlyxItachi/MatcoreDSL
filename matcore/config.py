"""MatCore global configuration system.

Priority order (highest to lowest):
1. Environment variables (MATCORE_TARGET, MATCORE_DEBUG, etc.)
2. Per-launch kwargs in mc.launch()
3. Global config set via mc.config()
4. Built-in defaults
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return None


@dataclass
class MatCoreConfig:
    """Global configuration for MatCore."""

    default_target: str = "x86-auto"
    debug: bool = False
    trace: str = "none"
    cache_dir: str = ""  # Empty = use default .matcore_cache
    validate: bool = False
    log_level: str = "warning"  # debug, info, warning, error
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        if env_target := os.environ.get("MATCORE_TARGET"):
            self.default_target = env_target
        env_debug = _env_bool("MATCORE_DEBUG")
        if env_debug is not None:
            self.debug = env_debug
        if env_trace := os.environ.get("MATCORE_TRACE"):
            self.trace = env_trace.lower()
        if env_cache := os.environ.get("MATCORE_CACHE_DIR"):
            self.cache_dir = env_cache
        env_validate = _env_bool("MATCORE_VALIDATE")
        if env_validate is not None:
            self.validate = env_validate
        if env_log := os.environ.get("MATCORE_LOG_LEVEL"):
            self.log_level = env_log.lower()


_global_config = MatCoreConfig()


def get_config() -> MatCoreConfig:
    """Get the current global configuration."""
    return deepcopy(_global_config)


def configure(**kwargs: Any) -> None:
    """Update global configuration.

    Example:
        mc.config(default_target="nvidia-dgpu:sm_90", debug=True)
    """
    global _global_config
    valid_keys = {f.name for f in _global_config.__dataclass_fields__.values()}
    for key, value in kwargs.items():
        if key not in valid_keys:
            raise ValueError(
                f"Unknown config key '{key}'. Valid keys: {', '.join(sorted(valid_keys))}"
            )
        setattr(_global_config, key, value)


def reset_config() -> None:
    """Reset configuration to built-in defaults."""
    global _global_config
    _global_config = MatCoreConfig()


def resolve_launch_options(
    *,
    target: str | None = None,
    debug: bool | None = None,
    trace: str | None = None,
    validate: bool | None = None,
) -> dict[str, Any]:
    """Resolve final options: env > launch kwargs > global config > defaults."""
    cfg = get_config()
    resolved_target = target if target is not None else cfg.default_target
    resolved_debug = debug if debug is not None else cfg.debug
    resolved_trace = trace if trace is not None else cfg.trace
    resolved_validate = validate if validate is not None else cfg.validate

    env_debug = _env_bool("MATCORE_DEBUG")
    if env_debug is not None:
        resolved_debug = env_debug
    if env_trace := os.environ.get("MATCORE_TRACE"):
        resolved_trace = env_trace.lower()
    if env_target := os.environ.get("MATCORE_TARGET"):
        resolved_target = env_target
    env_validate = _env_bool("MATCORE_VALIDATE")
    if env_validate is not None:
        resolved_validate = env_validate

    return {
        "target": resolved_target,
        "debug": resolved_debug,
        "trace": resolved_trace,
        "validate": resolved_validate,
    }
