"""Configuration loading utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import project_root, resolve_path


@lru_cache(maxsize=4)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load YAML configuration from ``config/config.yaml`` by default."""
    root = project_root()
    path = (
        resolve_path(config_path, root)
        if config_path
        else root / "config" / "config.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
