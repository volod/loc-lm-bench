"""Profile one LOCAL embedder end to end: load it, encode a corpus, record the host it ran on.

This is the only part of the throughput lane that touches a real model, a real device, and real
telemetry; the measurement arithmetic and the host reading are pure and live next door.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

from llb.rag.encoder_throughput import (
    DEFAULT_MAX_WARM_PASSES,
    DEFAULT_MAX_WARM_SECONDS,
    DEFAULT_MIN_WARM_PASSES,
    DEFAULT_TARGET_RELATIVE_PRECISION,
    ThroughputProfile,
    measure_encoder_throughput,
)

_LOG = logging.getLogger(__name__)


def profile_local_embedder(
    model: str,
    texts: Sequence[str],
    *,
    device: str,
    target_relative_precision: float = DEFAULT_TARGET_RELATIVE_PRECISION,
    min_warm_passes: int = DEFAULT_MIN_WARM_PASSES,
    max_warm_passes: int = DEFAULT_MAX_WARM_PASSES,
    max_warm_seconds: float = DEFAULT_MAX_WARM_SECONDS,
    vram_reader: Callable[[], int] | None = None,
    power_reader: Callable[[], float | None] | None = None,
) -> ThroughputProfile:
    """Production binder: `Embedder(model, device=...)` + host GPU/power samplers."""
    import time

    from llb.backends.telemetry_samplers import PowerSampler, VramSampler
    from llb.rag.embedding import Embedder

    embedder = Embedder(model, device=device if device != "auto" else None)
    notes = _backend_notes(device)
    host = _host_identity()

    with VramSampler(vram_reader) as vram, PowerSampler(power_reader) as power:
        if vram_reader is not None:
            vram.sample()
        if power_reader is not None:
            power.sample()

        def load() -> Any:
            return embedder._load()

        def encode(batch: Sequence[str]) -> Any:
            return embedder.encode_passages(list(batch))

        profile = measure_encoder_throughput(
            model,
            texts,
            load=load,
            encode=encode,
            device=device,
            clock=time.perf_counter,
            target_relative_precision=target_relative_precision,
            min_warm_passes=min_warm_passes,
            max_warm_passes=max_warm_passes,
            max_warm_seconds=max_warm_seconds,
            peak_vram_mb=float(vram.peak_mb) if vram.peak_mb else None,
            mean_power_w=(sum(power.samples) / len(power.samples) if power.samples else None),
            peak_power_w=max(power.samples) if power.samples else None,
            power_limit_w=host.get("power_limit_w"),
            gpu_name=host.get("gpu_name"),
            driver=host.get("driver"),
            backend_notes=notes,
        )
    # Drop the weights so the next candidate does not stack VRAM on a 12 GiB host.
    embedder.release()
    return profile


def _backend_notes(device: str) -> dict[str, Any]:
    """Best-effort torch / CUDA / cuDNN flags for kernel-fallback inspection."""
    notes: dict[str, Any] = {"requested_device": device}
    try:
        import torch

        notes["torch"] = torch.__version__
        notes["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            notes["cuda_device_name"] = torch.cuda.get_device_name(0)
            notes["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
            notes["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: None)()
            notes["tf32_matmul"] = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
            notes["tf32_cudnn"] = bool(getattr(torch.backends.cudnn, "allow_tf32", False))
    except Exception as exc:  # optional path; never fail the bake-off for notes
        notes["torch_error"] = str(exc)
        _LOG.debug("[encoder-throughput] backend notes unavailable: %s", exc)
    return notes


def _host_identity() -> dict[str, Any]:
    """GPU name / driver / power limit from nvidia-smi (best-effort)."""
    out: dict[str, Any] = {}
    try:
        from llb.backends.hardware import detect_gpus

        gpus = detect_gpus()
        if gpus:
            out["gpu_name"] = gpus[0].name
            out["driver"] = gpus[0].driver
    except Exception as exc:
        _LOG.debug("[encoder-throughput] gpu detect failed: %s", exc)
    try:
        import subprocess

        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines():
                token = line.strip().split()[0] if line.strip() else ""
                try:
                    out["power_limit_w"] = float(token)
                    break
                except ValueError:
                    continue
    except Exception as exc:
        _LOG.debug("[encoder-throughput] power.limit unavailable: %s", exc)
    return out
