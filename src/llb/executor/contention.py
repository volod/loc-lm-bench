"""Pre-launch VRAM-contention guard (VRAM contention guard).

vLLM's startup free-memory check requires `gpu-memory-utilization x total_vram` to be
AVAILABLE (free) on the device -- it cannot reserve memory another process already holds. The
first real-model validation launch failed because Ollama held ~2.8 GB resident. This guard runs before a
VRAM-owning backend (vLLM) starts: it reads the device's free VRAM + the resident users, then
AUTO-DERATES `gpu-memory-utilization` to the actually-free fraction (the non-destructive
default). `--evict` (unload Ollama) and `--wait` (poll until VRAM frees) are explicit opt-in and
never kill another process implicitly. If even the derated target cannot hold the model's weight
floor (the embedding-aware memory planner estimate) plus a minimal KV working set, it aborts with an
actionable message. The single-run analogue of the isolation reclaim cross-cell VRAM gate.

The free VRAM comes from nvidia-smi (so the derate works without the `[telemetry]` extra); the
resident-PID attribution uses NVML when present (best-effort). Readers, the Ollama evict, and
sleep are injectable, so the logic is unit-testable without a GPU.
"""

import math
import os
from typing import Callable

from llb.core.config_validation import DEFAULT_OLLAMA_HOST
from llb.core.contracts.hardware import ContentionReport, ResidentProc
from llb.executor.contention_memory import DEFAULT_MIN_KV_HEADROOM_MB, DEFAULT_VLLM_OVERHEAD_MB
from llb.executor.ollama_eviction import evict_ollama


# Headroom (MiB) kept free below the reserved fraction: vLLM needs a sliver outside its
# gpu-memory-utilization budget for the CUDA context, and free VRAM jitters a little.
DEFAULT_MARGIN_MB = 512
# vLLM reserves a large NON-weight working set inside its budget before any KV blocks: the CUDA
# context, peak activations, and CUDA-graph capture. Measured ~1.8-2.4 GB for an 8B model on the
# RTX 4060 Ti (VRAM contention guard live validation: at a 11793 MB budget, weights ~10035 MB left 0 for KV and
# vLLM aborted with "No available memory for the cache blocks"). The abort check must include it,
# or the guard derates into a doomed launch.
# Minimal KV working set beyond weights + overhead for the abort check (room for >=1 KV block).
# Tokens of KV the abort check requires a launch to be able to serve (VRAM contention guard): the arch-derived
# headroom is the KV cache at this context, so a model that cannot hold even this much is aborted.
DEFAULT_WAIT_TIMEOUT_S = 120.0
DEFAULT_WAIT_POLL_S = 3.0

ACTION_OK = "ok"  # requested gpu-memory-utilization already fits the free VRAM
ACTION_DERATE = "derate"  # lowered gpu-memory-utilization to the free fraction
ACTION_ABORT = "abort"  # even the derated target cannot hold the model

GpuReader = Callable[[], "tuple[int, int] | None"]  # -> (total_mb, free_mb)
ProcessReader = Callable[[], dict[int, int]]  # -> {pid: used_mb}


def _floor2(value: float) -> float:
    """Round DOWN to 2 decimals, so the derated utilization stays below the free fraction."""
    return math.floor(max(0.0, value) * 100) / 100


def resident_users(usage: dict[int, int], exclude: set[int] | None = None) -> list[ResidentProc]:
    """{pid: used_mb} -> the other GPU processes holding VRAM, biggest first."""
    skip = exclude or set()
    procs: list[ResidentProc] = [
        {"pid": pid, "used_mb": mb} for pid, mb in usage.items() if pid not in skip and mb > 0
    ]
    procs.sort(key=lambda p: p["used_mb"], reverse=True)
    return procs


def _residents_note(residents: list[ResidentProc]) -> str:
    if not residents:
        return "no other GPU process"
    head = residents[0]
    extra = f" (+{len(residents) - 1} more)" if len(residents) > 1 else ""
    return f"PID {head['pid']} holds {head['used_mb']} MB{extra}"


def plan_guard(
    total_mb: int,
    free_mb: int,
    requested_util: float,
    weight_floor_mb: float,
    residents: list[ResidentProc] | None = None,
    *,
    margin_mb: int = DEFAULT_MARGIN_MB,
    overhead_mb: int = DEFAULT_VLLM_OVERHEAD_MB,
    min_kv_headroom_mb: int = DEFAULT_MIN_KV_HEADROOM_MB,
) -> ContentionReport:
    """Decide the safe `gpu-memory-utilization` for the current free VRAM (pure).

    `safe_util` is the requested value capped at `(free - margin) / total` (rounded down). It is
    only ever lowered, never raised. `fits` is False -- an abort -- when the resulting target
    cannot hold the weight floor + vLLM's serving overhead + a minimal KV working set.
    """
    residents = residents or []
    usable_mb = max(0, free_mb - margin_mb)
    safe_util = max(
        0.0, min(requested_util, _floor2(usable_mb / total_mb if total_mb > 0 else 0.0))
    )
    target_mb = int(safe_util * total_mb)
    needed_mb = int(weight_floor_mb + overhead_mb + min_kv_headroom_mb)
    fits = safe_util > 0 and target_mb >= needed_mb
    derated = safe_util < requested_util - 1e-9

    if not fits:
        action = ACTION_ABORT
        note = (
            f"only {free_mb} MB free of {total_mb} MB VRAM; this model needs ~{needed_mb} MB "
            f"(weights {int(weight_floor_mb)} + ~{overhead_mb} serving overhead + KV). "
            f"{_residents_note(residents)}. Free VRAM first (--evict to unload Ollama, --wait to "
            f"let a job finish) or pick a smaller model."
        )
    elif derated:
        action = ACTION_DERATE
        note = (
            f"derated gpu-memory-utilization {requested_util:.2f} -> {safe_util:.2f} "
            f"({free_mb} MB free of {total_mb}; {_residents_note(residents)})"
        )
    else:
        action = ACTION_OK
        note = f"gpu-memory-utilization {requested_util:.2f} fits ({free_mb} MB free)"

    return {
        "total_mb": total_mb,
        "free_mb": free_mb,
        "requested_util": round(requested_util, 2),
        "safe_util": safe_util,
        "target_mb": target_mb,
        "weight_floor_mb": int(weight_floor_mb),
        "residents": residents,
        "derated": derated,
        "fits": fits,
        "action": action,
        "note": note,
    }


def apply_contention_guard(
    *,
    requested_util: float,
    weight_floor_mb: float,
    gpu_reader: GpuReader,
    process_reader: ProcessReader | None = None,
    own_pids: set[int] | None = None,
    evict: bool = False,
    wait: bool = False,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    evict_fn: Callable[[str], None] | None = None,
    wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    poll_s: float = DEFAULT_WAIT_POLL_S,
    sleep: Callable[[float], None] | None = None,
    margin_mb: int = DEFAULT_MARGIN_MB,
    overhead_mb: int = DEFAULT_VLLM_OVERHEAD_MB,
    min_kv_headroom_mb: int = DEFAULT_MIN_KV_HEADROOM_MB,
) -> ContentionReport | None:
    """Read the device, optionally free VRAM, then plan the derate. None when no GPU is present.

    `--evict` unloads Ollama's resident models; `--wait` polls until the requested target would
    fit (or the timeout). Both re-read free VRAM before planning. Neither ever kills a process.
    """
    gpu = gpu_reader()
    if gpu is None:
        return None  # no GPU -> nothing to guard (best-effort; the launch proceeds unchanged)
    total_mb, free_mb = gpu

    if evict:
        (evict_fn or evict_ollama)(ollama_host)
    if evict or wait:
        target_free = int(requested_util * total_mb) + margin_mb
        free_mb = _poll_free(
            gpu_reader, target_free, wait_timeout_s, poll_s, sleep, fallback=free_mb
        )

    residents: list[ResidentProc] = []
    if process_reader is not None:
        try:
            residents = resident_users(process_reader(), exclude=own_pids or {os.getpid()})
        except Exception:  # NVML hiccup -> just omit the attribution
            residents = []

    return plan_guard(
        total_mb,
        free_mb,
        requested_util,
        weight_floor_mb,
        residents,
        margin_mb=margin_mb,
        overhead_mb=overhead_mb,
        min_kv_headroom_mb=min_kv_headroom_mb,
    )


def _poll_free(
    gpu_reader: GpuReader,
    target_free_mb: int,
    timeout_s: float,
    poll_s: float,
    sleep: Callable[[float], None] | None,
    *,
    fallback: int,
) -> int:
    """Poll free VRAM until it reaches `target_free_mb` or the timeout; return the last reading."""
    import time

    nap = sleep or time.sleep

    def _free() -> int:
        gpu = gpu_reader()
        return gpu[1] if gpu else fallback

    free = _free()
    waited = 0.0
    while free < target_free_mb and waited < timeout_s:
        nap(poll_s)
        waited += poll_s
        free = _free()
    return free


# --- default readers + the Ollama evict (best-effort live IO) ------------------------------


def default_gpu_reader() -> tuple[int, int] | None:
    """(total_mb, free_mb) for the device the run TARGETS, via nvidia-smi (VRAM contention guard). None if no GPU.

    Reads ALL GPUs and selects the target (the `CUDA_VISIBLE_DEVICES` device when set, else the
    most-free GPU) instead of hard-coding GPU 0, so the guard is correct on a multi-GPU host."""
    from llb.backends.hardware import detect_gpus, select_target_gpu

    gpu = select_target_gpu(detect_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    return (gpu.total_mb, gpu.free_mb) if gpu is not None else None
