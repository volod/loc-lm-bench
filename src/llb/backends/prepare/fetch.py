"""The side-effecting fetchers: `ollama pull` and the Hugging Face snapshot download, plus the
gated-access / timeout / detail-formatting helpers around them.

Both `_ollama_pull` and `_hf_cache` are the injectable defaults `prepare_models` uses; tests pass
fakes instead so no CLI or network is touched.
"""

import os
import shutil
import subprocess
from pathlib import Path

from llb.backends.prepare.base import (
    DEFAULT_OLLAMA_PULL_TIMEOUT_S,
    MAX_ERROR_DETAIL_CHARS,
    OLLAMA_PULL_TIMEOUT_ENV,
)
from llb.core import env


def _looks_gated(exc: Exception) -> bool:
    """True if a download error is an access gate (license not accepted / no token)."""
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(
        s in blob
        for s in (
            "gated",
            "401",
            "403",
            "awaiting",
            "must be authenticated",
            "access to model",
            "accept the conditions",
        )
    )


def _ollama_timeout_s() -> int | None:
    raw = os.environ.get(OLLAMA_PULL_TIMEOUT_ENV)
    if raw is None or raw == "":
        return DEFAULT_OLLAMA_PULL_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_OLLAMA_PULL_TIMEOUT_S
    return value if value > 0 else None


def _ascii_detail(value: str) -> str:
    """Return a compact ASCII-only diagnostic string for captured tool output."""
    clean = value.encode("ascii", "ignore").decode("ascii")
    clean = " ".join(clean.split())
    if len(clean) > MAX_ERROR_DETAIL_CHARS:
        return "..." + clean[-MAX_ERROR_DETAIL_CHARS:]
    return clean


def _ollama_pull(source: str) -> tuple[bool, str]:
    if shutil.which("ollama") is None:
        return False, "ollama CLI not found (install Ollama and run `ollama serve`)"
    timeout_s = _ollama_timeout_s()
    try:
        out = subprocess.run(
            ["ollama", "pull", source],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        detail = f"ollama pull timed out after {timeout_s}s"
        return False, detail
    except subprocess.SubprocessError as exc:
        return False, f"ollama pull failed: {exc}"
    if out.returncode == 0:
        return True, "success"
    detail = _ascii_detail(f"{out.stderr}\n{out.stdout}")
    suffix = f": {detail}" if detail else ""
    return False, f"ollama pull exited with code {out.returncode}{suffix}"


def _hf_cache(source: str, token: str | None, cache_dir: Path | None) -> tuple[bool, str]:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import (
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError:
        return False, "huggingface_hub missing (it is a base dep; reinstall with `make venv`)"
    try:
        progress_was_disabled = are_progress_bars_disabled()
        disable_progress_bars()
        path = snapshot_download(
            repo_id=source,
            token=token or os.environ.get(env.HF_TOKEN),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:  # 404 / auth / network -- report per-model, keep going
        msg = f"{type(exc).__name__}: {exc}"
        if _looks_gated(exc):
            msg += (
                f" -- accept the license at https://huggingface.co/{source} "
                f"and set {env.HF_TOKEN} in .env"
            )
        return False, msg
    finally:
        if "progress_was_disabled" in locals() and not progress_was_disabled:
            enable_progress_bars()
    return True, path
