"""Pre-launch VRAM-contention guard (M4.2).

vLLM's startup free-memory check requires `gpu-memory-utilization x total_vram` to be
AVAILABLE (free) on the device -- it cannot reserve memory another process already holds. The
first M2.4 launch failed because Ollama held ~2.8 GB resident. This guard runs before a
VRAM-owning backend (vLLM) starts: it reads the device's free VRAM + the resident users, then
AUTO-DERATES `gpu-memory-utilization` to the actually-free fraction (the non-destructive
default). `--evict` (unload Ollama) and `--wait` (poll until VRAM frees) are explicit opt-in and
never kill another process implicitly. If even the derated target cannot hold the model's weight
floor (the embedding-aware M4.1 estimate) plus a minimal KV working set, it aborts with an
actionable message. The single-run analogue of the M3.3 cross-cell VRAM gate.

The free VRAM comes from nvidia-smi (so the derate works without the `[telemetry]` extra); the
resident-PID attribution uses NVML when present (best-effort). Readers, the Ollama evict, and
sleep are injectable, so the logic is unit-testable without a GPU.
"""

import json
import logging
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, cast

from llb.config import DEFAULT_OLLAMA_HOST
from llb.contracts import ContentionReport, ResidentProc

_LOG = logging.getLogger(__name__)

# Headroom (MiB) kept free below the reserved fraction: vLLM needs a sliver outside its
# gpu-memory-utilization budget for the CUDA context, and free VRAM jitters a little.
DEFAULT_MARGIN_MB = 512
# vLLM reserves a large NON-weight working set inside its budget before any KV blocks: the CUDA
# context, peak activations, and CUDA-graph capture. Measured ~1.8-2.4 GB for an 8B model on the
# RTX 4060 Ti (M4.2 live validation: at a 11793 MB budget, weights ~10035 MB left 0 for KV and
# vLLM aborted with "No available memory for the cache blocks"). The abort check must include it,
# or the guard derates into a doomed launch.
DEFAULT_VLLM_OVERHEAD_MB = 2048
# Minimal KV working set beyond weights + overhead for the abort check (room for >=1 KV block).
DEFAULT_MIN_KV_HEADROOM_MB = 512
# Tokens of KV the abort check requires a launch to be able to serve (M4.2): the arch-derived
# headroom is the KV cache at this context, so a model that cannot hold even this much is aborted.
DEFAULT_MIN_SERVING_CTX = 2048
DEFAULT_WAIT_TIMEOUT_S = 120.0
DEFAULT_WAIT_POLL_S = 3.0
DEFAULT_MANIFEST = Path("samples/models_uk.yaml")

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
    """(total_mb, free_mb) for the device the run TARGETS, via nvidia-smi (M4.2). None if no GPU.

    Reads ALL GPUs and selects the target (the `CUDA_VISIBLE_DEVICES` device when set, else the
    most-free GPU) instead of hard-coding GPU 0, so the guard is correct on a multi-GPU host."""
    from llb.backends.hardware import detect_gpus, select_target_gpu

    gpu = select_target_gpu(detect_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    return (gpu.total_mb, gpu.free_mb) if gpu is not None else None


def _spec_for(model_source: str, manifest: Path) -> "dict[str, Any] | None":
    from llb.backends.prepare import load_manifest

    for spec in load_manifest(manifest):
        if spec.get("source") == model_source or spec.get("name") == model_source:
            return cast("dict[str, Any]", spec)
    return None


def model_weight_floor_mb(model_source: str, manifest: Path = DEFAULT_MANIFEST) -> float:
    """Embedding-aware weights estimate (MiB, M4.1) for a model by source/name. 0.0 if unknown."""
    try:
        from llb.backends.planner import enrich_arch, plan_model

        spec = _spec_for(model_source, manifest)
        if spec is None:
            return 0.0
        row = plan_model(enrich_arch(cast(Any, spec)), vram_mib=1_000_000, ram_mib=1_000_000)
        return float(row["weights_mib"] or 0.0)
    except Exception:
        return 0.0


def model_kv_headroom_mb(
    model_source: str,
    manifest: Path = DEFAULT_MANIFEST,
    *,
    min_serving_ctx: int = DEFAULT_MIN_SERVING_CTX,
) -> int:
    """Arch-derived minimal KV working set (MiB, M4.2): the KV the model needs to serve at least
    `min_serving_ctx` tokens, sliding-window-aware. The abort check uses this instead of a fixed
    floor, so a model with a heavy KV per token is judged un-launchable at the right threshold.
    Falls back to the fixed floor when the arch is unknown."""
    try:
        from llb.backends.planner import enrich_arch, kv_mib_at_context

        spec = _spec_for(model_source, manifest)
        if spec is None:
            return DEFAULT_MIN_KV_HEADROOM_MB
        enriched = cast("dict[str, Any]", enrich_arch(cast(Any, spec)))
        n_layers = enriched.get("n_layers")
        kv_dim = enriched.get("kv_dim")
        if not (n_layers and kv_dim):
            return DEFAULT_MIN_KV_HEADROOM_MB
        kv = kv_mib_at_context(
            int(n_layers),
            int(kv_dim),
            min_serving_ctx,
            sliding_window=enriched.get("sliding_window"),
            sliding_window_pattern=enriched.get("sliding_window_pattern"),
        )
        return max(DEFAULT_MIN_KV_HEADROOM_MB, int(kv))
    except Exception:
        return DEFAULT_MIN_KV_HEADROOM_MB


def evict_ollama(
    host: str = DEFAULT_OLLAMA_HOST,
    *,
    http_get: Callable[[str], dict[str, Any] | None] | None = None,
    http_post: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """Ask Ollama to unload every resident model (`keep_alive: 0`). Best-effort; never raises."""
    get = http_get or _http_get_json
    post = http_post or _http_post_json
    base = host.rstrip("/")
    try:
        running = get(f"{base}/api/ps")
    except Exception:
        return
    for entry in (running or {}).get("models", []):
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        try:
            post(f"{base}/api/generate", {"model": name, "keep_alive": 0})
            _LOG.info("[contention] requested Ollama unload of %s (keep_alive=0)", name)
        except Exception:
            continue


def _http_get_json(url: str, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return cast(dict[str, Any], json.loads(resp.read().decode("utf-8", "replace")))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 10.0) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout):
        return
