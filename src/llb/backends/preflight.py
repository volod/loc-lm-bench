"""vLLM flashinfer sampler preflight (M4.3).

vLLM JIT-compiles flashinfer's sampling kernel at engine startup. On consumer sm_89 GPUs the
flashinfer 0.6.x `sampling.cuh` calls a `cub::BlockAdjacentDifference::FlagHeads` that newer
CCCL/CUB removed, so the build fails and the engine never comes up -- which is why the sampler is
OFF by default. This preflight runs the kernel build ONCE (during `build-vllm`), records a
DEFINITIVE verdict, and `launch_env` reads it: flashinfer is re-enabled only when the verdict says
the kernel builds on this host, else the safe native sampler stays selected.

The probe is injectable, so the verdict logic + persistence are unit-testable without CUDA; the
real build-once probe runs only on the CUDA host that `build-vllm` targets.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypedDict, cast

from llb.paths import resolve_data_dir

_LOG = logging.getLogger(__name__)

SAMPLER_FLASHINFER = "flashinfer"  # the JIT sampler builds + runs here -> enable it
SAMPLER_NATIVE = "native"  # build/run failed (or no flashinfer) -> vLLM's native sampler (safe)

# True == the flashinfer sampling kernel builds AND runs on this host.
SamplerProbe = Callable[[], bool]


class SamplerVerdict(TypedDict):
    sampler: str  # flashinfer | native
    flashinfer_version: str | None
    detail: str
    checked_at: str  # ISO-8601 UTC, for provenance


def verdict_path(data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else resolve_data_dir()
    return base / "llb" / "preflight" / "vllm_sampler.json"


def probe_sampler(probe: SamplerProbe | None = None) -> SamplerVerdict:
    """Run the flashinfer-sampler build probe and return a definitive verdict (never raises)."""
    runner = probe or _default_flashinfer_probe
    version = _flashinfer_version()
    try:
        ok = runner()
        detail = (
            "flashinfer sampling kernel built and ran"
            if ok
            else "flashinfer sampler unavailable; using vLLM's native sampler"
        )
    except Exception as exc:  # a broken JIT build must yield `native`, not crash the preflight
        ok = False
        detail = f"flashinfer sampler probe failed: {type(exc).__name__}: {exc}"
    return {
        "sampler": SAMPLER_FLASHINFER if ok else SAMPLER_NATIVE,
        "flashinfer_version": version,
        "detail": detail,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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


def run_preflight(
    *, probe: SamplerProbe | None = None, data_dir: Path | None = None
) -> SamplerVerdict:
    """Probe the flashinfer sampler, persist the verdict, and return it (the `build-vllm` hook)."""
    verdict = probe_sampler(probe)
    path = save_verdict(verdict, data_dir)
    _LOG.info(
        "[preflight] flashinfer sampler verdict: %s -- %s (recorded -> %s)",
        verdict["sampler"],
        verdict["detail"],
        path,
    )
    return verdict


def _flashinfer_version() -> str | None:
    import importlib.metadata as metadata

    for name in ("flashinfer-python", "flashinfer"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _default_flashinfer_probe() -> bool:
    """Best-effort: import flashinfer + JIT-build and run its sampling kernel on the GPU.

    CUDA-only and intentionally tolerant -- any import/build/runtime failure means the kernel
    does not work on this host, so the verdict is `native`. Injected with a fake in tests.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        from flashinfer.sampling import sampling_from_probs

        probs = torch.rand(4, 32, device="cuda")
        probs = probs / probs.sum(dim=-1, keepdim=True)
        sampling_from_probs(probs)  # forces the kernel JIT build + a real launch
        torch.cuda.synchronize()
        return True
    except Exception:
        return False
