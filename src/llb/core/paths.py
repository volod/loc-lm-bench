"""Project-relative path and environment resolution.

This module is the single source of truth for locating the repository and runtime data.
Relative paths are resolved against the project root, never against the caller's current
working directory. Environment variables from the project ``.env`` file are loaded without
overriding values already supplied by the process.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from llb.core import env

DEFAULT_DATA_DIR = ".data"
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/llb/core/paths.py -> repo root


def load_project_env() -> None:
    """Load the project ``.env`` file while preserving process environment overrides."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_project_path(path: Path | str) -> Path:
    """Return an absolute path, resolving relative input from the project root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolve_data_dir(value: Path | str | None = None) -> Path:
    """Resolve an explicit data root or ``DATA_DIR`` from the project environment."""
    load_project_env()
    configured = value if value is not None else os.environ.get(env.DATA_DIR, DEFAULT_DATA_DIR)
    return resolve_project_path(configured)
