"""Encoder throughput decomposition: cold load, first-pass compile+encode, steady warm encodes.

A one-pass bake-off conflates model load, first-kernel CUDA compilation, and steady encoding into
one `embed_seconds` number. On a 12 GiB Blackwell laptop that mix produced a much wider
architecture-dependent spread than a workstation run, so a host recommendation that reads those
rates as model properties is wrong. This module separates the three phases, repeats the warm
encode until a relative-precision target or a resource cap, and reports whether the one-pass rate
ordering survives the warm measurement.
"""

import logging
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from typing_extensions import NotRequired, TypedDict

_LOG = logging.getLogger(__name__)

# Stop when the warm-pass IQR is this share of the median (or smaller).
DEFAULT_TARGET_RELATIVE_PRECISION = 0.05
DEFAULT_MIN_WARM_PASSES = 3
DEFAULT_MAX_WARM_PASSES = 10
# Wall-clock cap over the warm loop only (load + first pass are outside it).
DEFAULT_MAX_WARM_SECONDS = 180.0

STOP_PRECISION = "precision"
STOP_MAX_PASSES = "max_passes"
STOP_MAX_SECONDS = "max_seconds"

EncodeFn = Callable[[Sequence[str]], Any]
Clock = Callable[[], float]
Loader = Callable[[], Any]


class ThroughputProfile(TypedDict):
    """One encoder's decomposed timing on a fixed text set."""

    model: str
    n_texts: int
    device: str
    load_seconds: float
    first_pass_seconds: float
    compile_estimate_seconds: float
    warm_seconds: list[float]
    warm_median_seconds: float
    warm_iqr_seconds: float
    warm_relative_precision: float
    warm_passes: int
    stopping_reason: str
    target_relative_precision: float
    max_warm_passes: int
    max_warm_seconds: float
    warm_chunks_per_s: float
    one_pass_chunks_per_s: float
    peak_vram_mb: NotRequired[float | None]
    mean_power_w: NotRequired[float | None]
    peak_power_w: NotRequired[float | None]
    power_limit_w: NotRequired[float | None]
    gpu_name: NotRequired[str | None]
    driver: NotRequired[str | None]
    backend_notes: NotRequired[dict[str, Any]]


class FasterThanBaseline(TypedDict):
    """One encoder that beats the bake-off baseline on warm chunks/s (headline device)."""

    model: str
    warm_chunks_per_s: float
    speedup_vs_baseline: float | None
    peak_vram_mb: float | None


class HostThroughputSummary(TypedDict):
    """Cross-encoder host summary: warm ordering vs one-pass ordering."""

    corpus_n_texts: int
    devices: list[str]
    profiles: list[ThroughputProfile]
    # Per-device orderings so a CPU twin never reshuffles the CUDA recommendation.
    by_device: dict[str, dict[str, Any]]
    one_pass_order: list[str]
    warm_order: list[str]
    ordering_survives: bool
    verdict: str
    # When the bake-off baseline is known, list headline-device models that beat it on warm rate.
    baseline_model: NotRequired[str | None]
    faster_than_baseline: NotRequired[list[FasterThanBaseline]]


def relative_precision(samples: Sequence[float]) -> tuple[float, float, float]:
    """Return (median, iqr, iqr/median) over `samples`; zeros when empty or median is 0."""
    if not samples:
        return 0.0, 0.0, 0.0
    ordered = sorted(samples)
    median = statistics.median(ordered)
    if len(ordered) == 1:
        return float(median), 0.0, 0.0
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[mid:] if len(ordered) % 2 == 0 else ordered[mid + 1 :]
    iqr = float(statistics.median(upper) - statistics.median(lower)) if lower and upper else 0.0
    precision = (iqr / median) if median > 0 else 0.0
    return float(median), iqr, precision


def _check_measurement_contract(
    texts: Sequence[str],
    *,
    min_warm_passes: int,
    max_warm_passes: int,
    target_relative_precision: float,
) -> None:
    """Refuse a measurement whose stopping rule could never be satisfied."""
    if not texts:
        raise ValueError("encoder throughput needs a non-empty text set")
    if min_warm_passes < 1:
        raise ValueError("min_warm_passes must be >= 1")
    if max_warm_passes < min_warm_passes:
        raise ValueError("max_warm_passes must be >= min_warm_passes")
    if target_relative_precision <= 0:
        raise ValueError("target_relative_precision must be > 0")


def _warm_passes(
    texts: Sequence[str],
    *,
    encode: EncodeFn,
    clock: Clock,
    target_relative_precision: float,
    min_warm_passes: int,
    max_warm_passes: int,
    max_warm_seconds: float,
) -> tuple[list[float], str]:
    """Encode until the rate is precise enough, saying WHY the loop stopped.

    Adaptive rather than a fixed count: a stable encoder reaches its target in a few passes, and a
    noisy one is stopped by the pass or wall-clock budget instead of quoting a number it never
    resolved.
    """
    warm_seconds: list[float] = []
    budget_start = clock()
    for pass_index in range(1, max_warm_passes + 1):
        started = clock()
        encode(texts)
        ended = clock()
        warm_seconds.append(ended - started)
        _median, _iqr, precision = relative_precision(warm_seconds)
        if pass_index >= min_warm_passes and precision <= target_relative_precision:
            return warm_seconds, STOP_PRECISION
        if ended - budget_start >= max_warm_seconds:
            return warm_seconds, STOP_MAX_SECONDS
    return warm_seconds, STOP_MAX_PASSES


def _host_fields(
    *,
    peak_vram_mb: float | None,
    mean_power_w: float | None,
    peak_power_w: float | None,
    power_limit_w: float | None,
    gpu_name: str | None,
    driver: str | None,
    backend_notes: dict[str, Any] | None,
) -> dict[str, Any]:
    """The host-specific fields a profile carries only when the host actually reported them."""
    fields = {
        "peak_vram_mb": peak_vram_mb,
        "mean_power_w": mean_power_w,
        "peak_power_w": peak_power_w,
        "power_limit_w": power_limit_w,
        "gpu_name": gpu_name,
        "driver": driver,
        "backend_notes": backend_notes,
    }
    return {key: value for key, value in fields.items() if value is not None}


def measure_encoder_throughput(
    model: str,
    texts: Sequence[str],
    *,
    load: Loader,
    encode: EncodeFn,
    device: str,
    clock: Clock,
    target_relative_precision: float = DEFAULT_TARGET_RELATIVE_PRECISION,
    min_warm_passes: int = DEFAULT_MIN_WARM_PASSES,
    max_warm_passes: int = DEFAULT_MAX_WARM_PASSES,
    max_warm_seconds: float = DEFAULT_MAX_WARM_SECONDS,
    peak_vram_mb: float | None = None,
    mean_power_w: float | None = None,
    peak_power_w: float | None = None,
    power_limit_w: float | None = None,
    gpu_name: str | None = None,
    driver: str | None = None,
    backend_notes: dict[str, Any] | None = None,
) -> ThroughputProfile:
    """Cold-load, first-pass (compile+encode), then adaptive warm encodes over fixed `texts`.

    `load` / `encode` / `clock` are injectable so CI drives the arithmetic with a fake clock and
    fake encoder; the production path binds them to an `Embedder` and `time.perf_counter`.
    """
    _check_measurement_contract(
        texts,
        min_warm_passes=min_warm_passes,
        max_warm_passes=max_warm_passes,
        target_relative_precision=target_relative_precision,
    )
    t0 = clock()
    load()
    load_seconds = clock() - t0

    t1 = clock()
    encode(texts)
    first_pass_seconds = clock() - t1

    warm_seconds, stopping_reason = _warm_passes(
        texts,
        encode=encode,
        clock=clock,
        target_relative_precision=target_relative_precision,
        min_warm_passes=min_warm_passes,
        max_warm_passes=max_warm_passes,
        max_warm_seconds=max_warm_seconds,
    )
    median, iqr, precision = relative_precision(warm_seconds)
    n_texts = len(texts)
    one_pass = load_seconds + first_pass_seconds
    profile: ThroughputProfile = {
        "model": model,
        "n_texts": n_texts,
        "device": device,
        "load_seconds": round(load_seconds, 4),
        "first_pass_seconds": round(first_pass_seconds, 4),
        "compile_estimate_seconds": round(max(0.0, first_pass_seconds - median), 4),
        "warm_seconds": [round(value, 4) for value in warm_seconds],
        "warm_median_seconds": round(median, 4),
        "warm_iqr_seconds": round(iqr, 4),
        "warm_relative_precision": round(precision, 4),
        "warm_passes": len(warm_seconds),
        "stopping_reason": stopping_reason,
        "target_relative_precision": target_relative_precision,
        "max_warm_passes": max_warm_passes,
        "max_warm_seconds": max_warm_seconds,
        "warm_chunks_per_s": round(n_texts / median, 3) if median > 0 else 0.0,
        "one_pass_chunks_per_s": round(n_texts / one_pass, 3) if one_pass > 0 else 0.0,
    }
    profile.update(
        _host_fields(  # type: ignore[typeddict-item]
            peak_vram_mb=peak_vram_mb,
            mean_power_w=mean_power_w,
            peak_power_w=peak_power_w,
            power_limit_w=power_limit_w,
            gpu_name=gpu_name,
            driver=driver,
            backend_notes=backend_notes,
        )
    )
    return profile
