"""Focused preflight verdict implementation."""

import json
from pathlib import Path
from typing import TypedDict, cast
from llb.core.paths import resolve_data_dir

SAMPLER_FLASHINFER = "flashinfer"  # the JIT sampler builds + runs here -> enable it

SAMPLER_NATIVE = "native"  # build/run failed (or no flashinfer) -> vLLM's native sampler (safe)


class SamplerVerdict(TypedDict):
    sampler: str  # flashinfer | native
    flashinfer_version: str | None
    detail: str
    checked_at: str  # ISO-8601 UTC, for provenance
    driver: (
        str | None
    )  # GPU driver at probe time -- a change re-runs the preflight (vLLM serving preflight)
    pinned_version: str | None  # flashinfer version auto-pinned to make the sampler work, or None
    auto_pinned: bool  # True when a candidate flashinfer was installed to enable the sampler


def verdict_path(data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else resolve_data_dir()
    return base / "llb" / "preflight" / "vllm_sampler.json"


def verdict_is_current(verdict: SamplerVerdict | None, driver: str | None) -> bool:
    """True when a verdict exists AND was recorded under the current driver (vLLM serving preflight): a driver change
    invalidates the cached verdict, so the preflight re-runs WITHOUT a full vLLM rebuild."""
    if verdict is None:
        return False
    recorded = verdict.get("driver")
    return recorded is None or driver is None or recorded == driver


def save_verdict(verdict: SamplerVerdict, data_dir: Path | None = None) -> Path:
    path = verdict_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return path


def load_verdict(data_dir: Path | None = None) -> SamplerVerdict | None:
    """The persisted preflight verdict, or None when no preflight has run (best-effort)."""
    path = verdict_path(data_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and data.get("sampler") in (SAMPLER_FLASHINFER, SAMPLER_NATIVE):
        return cast(SamplerVerdict, data)
    return None


def flashinfer_sampler_ok(data_dir: Path | None = None) -> bool:
    """True only when a saved preflight verdict confirms the flashinfer sampler builds here."""
    verdict = load_verdict(data_dir)
    return verdict is not None and verdict["sampler"] == SAMPLER_FLASHINFER
