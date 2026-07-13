"""Backend stores, reuse detection, and the download disk-preflight.

Locates where each backend keeps its artifacts (Ollama blob store vs the HF hub cache), answers the
REUSE signal (is this artifact already cached, so prepare fetches ~nothing?), estimates the fetch
size, and decides whether a store filesystem has room for it.
"""

import os
from pathlib import Path

from llb.backends import planner
from llb.backends.prepare.base import (
    DOWNLOAD_HEADROOM_MB,
    DOWNLOAD_SAFETY_FACTOR,
    MIN_DOWNLOAD_MB,
)
from llb.core import env
from llb.core.config import DEFAULT_OLLAMA_HOST
from llb.core.contracts import ModelSpec


def estimate_download_mb(spec: ModelSpec) -> int:
    """Best-effort size (MB) of the artifact `prepare` would fetch for this spec.

    Prices the weights with the embedding-aware planner (so an fp8 checkpoint whose big vocab
    embedding stays bf16 is not under-counted) plus a safety factor for the tokenizer, index, and
    packaging files. Falls back to `min_vram_gb`, then a small floor, when the spec carries no
    size hints -- an unsized entry never blocks on a bogus estimate.
    """
    bpw = planner.resolve_bpw(spec)
    params_b = spec.get("params_b")
    if bpw is not None and params_b is not None:
        weights = planner.weights_mib_detailed(
            float(params_b), bpw, planner.hi_precision_params(spec)
        )
        return max(MIN_DOWNLOAD_MB, int(weights * DOWNLOAD_SAFETY_FACTOR))
    floor = int(spec.get("min_vram_gb", 0)) * 1024
    return max(MIN_DOWNLOAD_MB, floor)


# Ollama's blob store moves with the install: the user default, the systemd-package location, and
# the OLLAMA_MODELS override. Both the reuse check and the free-space probe must look where the
# running daemon actually writes, or a system-service install (store under /usr/share/ollama) makes
# every model look un-cached and its filesystem un-probed.
_OLLAMA_STORE_CANDIDATES = (
    Path.home() / ".ollama" / "models",
    Path("/usr/share/ollama/.ollama/models"),
    Path("/var/lib/ollama/models"),
)


def _ollama_candidate_stores() -> list[Path]:
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return [Path(override).expanduser()]
    return list(_OLLAMA_STORE_CANDIDATES)


def _ollama_store_dir() -> Path:
    """The Ollama blob store a pull lands in: the first candidate that already exists, else the
    user-home default (created on first pull). Honors OLLAMA_MODELS over the install locations."""
    candidates = _ollama_candidate_stores()
    return next((c for c in candidates if c.exists()), candidates[0])


def _hf_cache_dir(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def store_dir_for(backend: str, cache_dir: Path | None) -> Path:
    """The on-disk store a prepared artifact lands in: Ollama's blob store or the HF hub cache."""
    return _ollama_store_dir() if backend == "ollama" else _hf_cache_dir(cache_dir)


def _ollama_manifest_exists(store: Path, source: str) -> bool:
    """Offline fallback: is an Ollama tag already pulled into this blob store? (reuse signal)."""
    manifests = store / "manifests"
    if not manifests.is_dir():
        return False
    repo, _, tag = source.partition(":")
    tag = (tag or "latest").lower()
    leaf = repo.rstrip("/").split("/")[-1].lower()
    try:
        for hit in manifests.rglob("*"):
            if hit.is_file() and hit.name.lower() == tag and leaf in hit.parent.as_posix().lower():
                return True
    except OSError:
        return False
    return False


def _ollama_present(source: str) -> bool:
    """Is this Ollama tag already available, so a pull fetches ~nothing? (reuse signal).

    Authoritative path: ask the RUNNING daemon what it serves (the same `/api/tags` probe the
    resolver uses), so a tag pulled into any store the daemon is configured with counts -- not just
    the on-disk locations we guess. Falls back to scanning the candidate blob stores when the daemon
    is unreachable (offline), so the check never blocks a re-pull of an already-cached tag.
    """
    from llb.backends.resolver import _make_ollama_probe

    host = os.environ.get(env.OLLAMA_HOST) or DEFAULT_OLLAMA_HOST
    try:
        if _make_ollama_probe(host)(source):
            return True
    except Exception:
        pass
    return any(_ollama_manifest_exists(store, source) for store in _ollama_candidate_stores())


def _default_present_check(spec: ModelSpec) -> bool:
    """Reuse signal: is this artifact already in its backend store, so prepare fetches ~nothing?"""
    backend = spec.get("backend")
    source = str(spec.get("source", ""))
    if backend == "vllm":
        return planner.cached_config_path(source) is not None
    if backend == "ollama":
        return _ollama_present(source)
    return False


def disk_precheck(
    required_mb: int, free_mb: int, headroom_mb: int = DOWNLOAD_HEADROOM_MB
) -> tuple[bool, str]:
    """(ok, reason) for a download of `required_mb` onto a store with `free_mb` free.

    `free_mb <= 0` means the free space is UNKNOWN (probe failed), which never blocks -- the
    check only refuses when it can prove the filesystem is too small.
    """
    if free_mb <= 0:
        return True, ""
    need = required_mb + headroom_mb
    if free_mb >= need:
        return True, ""
    return False, (
        f"insufficient disk: need ~{need} MB (artifact ~{required_mb} MB + {headroom_mb} MB "
        f"headroom), only {free_mb} MB free -- free space or move the store before retrying"
    )
