"""vLLM flashinfer sampler preflight (vLLM serving preflight).

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
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypedDict, cast

from llb import env
from llb.paths import resolve_data_dir

_LOG = logging.getLogger(__name__)

SAMPLER_FLASHINFER = "flashinfer"  # the JIT sampler builds + runs here -> enable it
SAMPLER_NATIVE = "native"  # build/run failed (or no flashinfer) -> vLLM's native sampler (safe)

# Candidate flashinfer versions to try when the bundled one fails the probe (vLLM serving preflight auto-pin), in
# order. A starting list of releases that predate the CCCL/CUB break; override with
# LLB_FLASHINFER_CANDIDATES (comma-separated) for a host whose working version differs.
DEFAULT_FLASHINFER_CANDIDATES = ("0.2.5", "0.1.6")

# True == the flashinfer sampling kernel builds AND runs on this host.
SamplerProbe = Callable[[], bool]
# install a flashinfer version -> success (injectable so the auto-pin is testable without pip/CUDA)
FlashinferInstaller = Callable[[str], bool]


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


def _run_probe(runner: SamplerProbe) -> tuple[bool, str]:
    try:
        ok = runner()
        detail = (
            "flashinfer sampling kernel built and ran"
            if ok
            else "flashinfer sampler unavailable; using vLLM's native sampler"
        )
    except Exception as exc:  # a broken JIT build must yield `native`, not crash the preflight
        return False, f"flashinfer sampler probe failed: {type(exc).__name__}: {exc}"
    return ok, detail


def probe_sampler(
    probe: SamplerProbe | None = None,
    *,
    driver: str | None = None,
    candidates: tuple[str, ...] | None = None,
    installer: FlashinferInstaller | None = None,
) -> SamplerVerdict:
    """Run the flashinfer-sampler build probe and return a definitive verdict (never raises).

    When the bundled flashinfer fails AND `candidates` are given, try to AUTO-PIN a host-compatible
    version (install + re-probe each candidate in order, vLLM serving preflight); the pinned version is recorded.
    """
    runner = probe or _default_flashinfer_probe
    ok, detail = _run_probe(runner)
    pinned: str | None = None
    auto_pinned = False
    if not ok and candidates:
        pinned, ok = auto_pin_flashinfer(candidates, probe=runner, installer=installer)
        if ok:
            auto_pinned = True
            detail = f"flashinfer sampler enabled by auto-pinning flashinfer=={pinned}"
    return {
        "sampler": SAMPLER_FLASHINFER if ok else SAMPLER_NATIVE,
        "flashinfer_version": pinned or _flashinfer_version(),
        "detail": detail,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "driver": driver if driver is not None else current_driver(),
        "pinned_version": pinned,
        "auto_pinned": auto_pinned,
    }


def auto_pin_flashinfer(
    candidates: tuple[str, ...],
    *,
    probe: SamplerProbe | None = None,
    installer: FlashinferInstaller | None = None,
) -> tuple[str | None, bool]:
    """Install + verify a host-compatible flashinfer from `candidates` in order (vLLM serving preflight).

    Returns (pinned_version, ok). Each install is injectable so the auto-pin is testable without
    pip / CUDA; a candidate that installs but still fails the probe is skipped."""
    runner = probe or _default_flashinfer_probe
    install = installer or _pip_install_flashinfer
    for version in candidates:
        try:
            if install(version) and runner():
                _LOG.info("[preflight] auto-pinned flashinfer==%s (sampler builds + runs)", version)
                return version, True
        except Exception:
            continue
    return None, False


def configured_candidates() -> tuple[str, ...]:
    """Auto-pin candidate versions: LLB_FLASHINFER_CANDIDATES (comma-separated) or the default."""
    raw = os.environ.get(env.FLASHINFER_CANDIDATES)
    if raw:
        return tuple(v.strip() for v in raw.split(",") if v.strip())
    return DEFAULT_FLASHINFER_CANDIDATES


def current_driver() -> str | None:
    """The host GPU driver version (best-effort), recorded so a driver change re-runs the probe."""
    try:
        from llb.backends.hardware import detect_gpus

        gpus = detect_gpus()
        return gpus[0].driver if gpus else None
    except Exception:
        return None


def verdict_is_current(verdict: SamplerVerdict | None, driver: str | None) -> bool:
    """True when a verdict exists AND was recorded under the current driver (vLLM serving preflight): a driver change
    invalidates the cached verdict, so the preflight re-runs WITHOUT a full vLLM rebuild."""
    if verdict is None:
        return False
    recorded = verdict.get("driver")
    return recorded is None or driver is None or recorded == driver


def _pip_install_flashinfer(version: str) -> bool:
    """Install a pinned flashinfer-python (best-effort; runs only on the real preflight host)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"flashinfer-python=={version}"],
            capture_output=True,
            text=True,
            timeout=900,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


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
    *,
    probe: SamplerProbe | None = None,
    data_dir: Path | None = None,
    candidates: tuple[str, ...] | None = None,
    installer: FlashinferInstaller | None = None,
    driver: str | None = None,
    force: bool = False,
) -> SamplerVerdict:
    """Probe the flashinfer sampler, persist the verdict, and return it (the `build-vllm` hook).

    Idempotent on a stable host: a cached verdict recorded under the CURRENT driver is reused, so
    re-running is cheap. A DRIVER CHANGE (or `force`) re-runs the probe WITHOUT a full vLLM rebuild
    (vLLM serving preflight). When the bundled flashinfer fails, `candidates` are auto-pinned (install + re-probe)."""
    drv = driver if driver is not None else current_driver()
    if not force:
        existing = load_verdict(data_dir)
        if existing is not None and verdict_is_current(existing, drv):
            _LOG.info(
                "[preflight] cached verdict current for driver %s: %s (use --force to re-run)",
                drv,
                existing["sampler"],
            )
            return existing
    # Auto-pin is OPT-IN: only attempt pip installs of candidate flashinfer versions when the
    # caller passes a (possibly empty) candidate list. The default never touches the environment.
    cand = candidates if candidates is not None else ()
    verdict = probe_sampler(probe, driver=drv, candidates=cand, installer=installer)
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
