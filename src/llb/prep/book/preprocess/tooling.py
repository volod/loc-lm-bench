"""Which shortlisted dewarper is installed, without importing it or any image library."""

from pathlib import Path

from llb.core.paths import resolve_data_dir
from llb.prep.book.survey.envs import installed_env


def page_dewarp_version(data_dir: Path | None = None) -> str | None:
    """The pinned page-dewarp version in the light environment, if that environment exists."""
    return _pinned(data_dir, "light", "page-dewarp")


def uvdoc_version(data_dir: Path | None = None) -> str | None:
    """The pinned PaddleOCR version in the paddle environment, if that environment exists."""
    return _pinned(data_dir, "paddle", "paddleocr")


def _pinned(data_dir: Path | None, env_name: str, distribution: str) -> str | None:
    root = data_dir if data_dir is not None else resolve_data_dir()
    env = installed_env(root, env_name)
    if env is None:
        return None
    if not env.freeze.is_file():
        return "installed"
    token = distribution.lower().replace("-", "_")
    for line in env.freeze.read_text(encoding="utf-8", errors="replace").splitlines():
        name, separator, version = line.partition("==")
        if separator and name.lower().replace("-", "_") == token and version:
            return version
    return "installed"
