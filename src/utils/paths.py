"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or project_root()) / p
