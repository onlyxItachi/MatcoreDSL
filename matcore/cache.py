"""MatCore compilation cache management."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

_CACHE_DIR_NAME = ".matcore_cache"
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent.parent / _CACHE_DIR_NAME)


def _default_cache_root() -> Path:
    env_override = os.getenv("MATCORE_CACHE_DIR")
    if env_override is not None and env_override != "":
        return Path(env_override)
    return Path(DEFAULT_CACHE_DIR)


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _find_cache_dirs(base_dir: str | None = None, include_prefixed: bool = False) -> list[Path]:
    """Find MatCore cache directories using exact-match behavior by default."""
    if base_dir is None:
        default_dir = _default_cache_root()
        if include_prefixed:
            return sorted(
                item
                for item in _safe_iterdir(default_dir.parent)
                if _safe_is_dir(item) and item.name.startswith(_CACHE_DIR_NAME)
            )
        return [default_dir] if default_dir.is_dir() else []

    base = Path(base_dir)
    if not base.exists():
        return []
    if base.is_dir() and base.name == _CACHE_DIR_NAME:
        return [base]
    if include_prefixed:
        return sorted(
            item
            for item in _safe_iterdir(base)
            if _safe_is_dir(item) and item.name.startswith(_CACHE_DIR_NAME)
        )
    exact = base / _CACHE_DIR_NAME
    return [exact] if exact.is_dir() else []


def cache_info(base_dir: str | None = None, include_prefixed: bool = False) -> list[dict[str, Any]]:
    """List all cached kernel artifacts with metadata."""
    results = []
    for cache_dir in _find_cache_dirs(base_dir, include_prefixed=include_prefixed):
        entry: dict[str, Any] = {
            "cache_dir": str(cache_dir),
            "artifact_dirs": [],
            "total_size_bytes": 0,
            "artifact_count": 0,
        }
        for sub in _safe_iterdir(cache_dir):
            if not _safe_is_dir(sub):
                continue
            artifact: dict[str, Any] = {"name": sub.name, "files": [], "metadata": None}
            for f in _safe_iterdir(sub):
                size = 0
                try:
                    if f.is_file():
                        size = f.stat().st_size
                except OSError:
                    size = 0
                artifact["files"].append({"name": f.name, "size": size})
                entry["total_size_bytes"] += size
            metadata_path = sub / "metadata.json"
            try:
                has_metadata = metadata_path.exists()
            except OSError:
                has_metadata = False
            if has_metadata:
                try:
                    artifact["metadata"] = json.loads(metadata_path.read_text())
                except Exception:
                    artifact["metadata"] = None
            entry["artifact_dirs"].append(artifact)
            entry["artifact_count"] += 1
        results.append(entry)
    return results


def cache_clear(base_dir: str | None = None, include_prefixed: bool = False) -> int:
    """Delete cached artifacts. Returns number of directories removed."""
    count = 0
    for cache_dir in _find_cache_dirs(base_dir, include_prefixed=include_prefixed):
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            count += 1
        except OSError:
            continue
    return count


def cache_summary(base_dir: str | None = None, include_prefixed: bool = False) -> str:
    """Return a human-readable cache summary."""
    info = cache_info(base_dir, include_prefixed=include_prefixed)
    if not info:
        return "No MatCore cache directories found."
    total_artifacts = sum(e["artifact_count"] for e in info)
    total_bytes = sum(e["total_size_bytes"] for e in info)
    size_mb = total_bytes / (1024 * 1024)
    lines = [
        f"MatCore Cache Summary: {len(info)} cache dir(s), "
        f"{total_artifacts} artifact(s), {size_mb:.2f} MB total",
    ]
    for entry in info:
        lines.append(f"  {entry['cache_dir']}: {entry['artifact_count']} artifact(s)")
    return "\n".join(lines)
