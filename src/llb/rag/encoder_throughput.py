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
    if not texts:
        raise ValueError("encoder throughput needs a non-empty text set")
    if min_warm_passes < 1:
        raise ValueError("min_warm_passes must be >= 1")
    if max_warm_passes < min_warm_passes:
        raise ValueError("max_warm_passes must be >= min_warm_passes")
    if target_relative_precision <= 0:
        raise ValueError("target_relative_precision must be > 0")

    t0 = clock()
    load()
    load_seconds = clock() - t0

    t1 = clock()
    encode(texts)
    first_pass_seconds = clock() - t1

    warm_seconds: list[float] = []
    warm_budget_start = clock()
    stopping_reason = STOP_MAX_PASSES
    for pass_i in range(1, max_warm_passes + 1):
        started = clock()
        encode(texts)
        ended = clock()
        warm_seconds.append(ended - started)
        _median, _iqr, precision = relative_precision(warm_seconds)
        if pass_i >= min_warm_passes and precision <= target_relative_precision:
            stopping_reason = STOP_PRECISION
            break
        if ended - warm_budget_start >= max_warm_seconds:
            stopping_reason = STOP_MAX_SECONDS
            break

    median, iqr, precision = relative_precision(warm_seconds)
    n_texts = len(texts)
    compile_estimate = max(0.0, first_pass_seconds - median)
    one_pass = load_seconds + first_pass_seconds
    profile: ThroughputProfile = {
        "model": model,
        "n_texts": n_texts,
        "device": device,
        "load_seconds": round(load_seconds, 4),
        "first_pass_seconds": round(first_pass_seconds, 4),
        "compile_estimate_seconds": round(compile_estimate, 4),
        "warm_seconds": [round(s, 4) for s in warm_seconds],
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
    if peak_vram_mb is not None:
        profile["peak_vram_mb"] = peak_vram_mb
    if mean_power_w is not None:
        profile["mean_power_w"] = mean_power_w
    if peak_power_w is not None:
        profile["peak_power_w"] = peak_power_w
    if power_limit_w is not None:
        profile["power_limit_w"] = power_limit_w
    if gpu_name is not None:
        profile["gpu_name"] = gpu_name
    if driver is not None:
        profile["driver"] = driver
    if backend_notes is not None:
        profile["backend_notes"] = backend_notes
    return profile


def rate_order(profiles: Sequence[ThroughputProfile], *, warm: bool) -> list[str]:
    """Models sorted by chunks/s descending (warm median or one-pass), ties by model id."""
    key = "warm_chunks_per_s" if warm else "one_pass_chunks_per_s"

    def sort_key(profile: ThroughputProfile) -> tuple[float, str]:
        return (-float(profile[key]), profile["model"])  # type: ignore[literal-required]

    return [profile["model"] for profile in sorted(profiles, key=sort_key)]


def ordering_survives(profiles: Sequence[ThroughputProfile]) -> bool:
    """True when one-pass and warm rate rankings name the same model sequence."""
    return rate_order(profiles, warm=False) == rate_order(profiles, warm=True)


def models_faster_than_baseline(
    profiles: Sequence[ThroughputProfile],
    *,
    baseline: str,
    device: str,
) -> list[FasterThanBaseline]:
    """Headline-device encoders whose warm chunks/s beat `baseline` (excluding the baseline)."""
    group = [p for p in profiles if p["device"] == device]
    base = next((p for p in group if p["model"] == baseline), None)
    if base is None:
        return []
    base_rate = float(base["warm_chunks_per_s"])
    rows: list[FasterThanBaseline] = []
    for profile in group:
        if profile["model"] == baseline:
            continue
        rate = float(profile["warm_chunks_per_s"])
        if rate <= base_rate:
            continue
        peak = profile.get("peak_vram_mb")
        rows.append(
            {
                "model": profile["model"],
                "warm_chunks_per_s": rate,
                "speedup_vs_baseline": (rate / base_rate) if base_rate > 0 else None,
                "peak_vram_mb": float(peak) if isinstance(peak, int | float) else None,
            }
        )
    rows.sort(key=lambda row: (-row["warm_chunks_per_s"], row["model"]))
    return rows


def build_host_summary(
    profiles: Sequence[ThroughputProfile],
    *,
    corpus_n_texts: int,
    baseline_model: str | None = None,
) -> HostThroughputSummary:
    """Cross-encoder summary: per-device orders, survival flag, and a one-line verdict."""
    devices = sorted({p["device"] for p in profiles})
    by_device: dict[str, dict[str, Any]] = {}
    for device in devices:
        group = [p for p in profiles if p["device"] == device]
        one_pass = rate_order(group, warm=False)
        warm = rate_order(group, warm=True)
        by_device[device] = {
            "one_pass_order": one_pass,
            "warm_order": warm,
            "ordering_survives": one_pass == warm,
        }

    # Headline order prefers CUDA when present (the host recommendation device).
    headline_device = "cuda" if "cuda" in by_device else (devices[0] if devices else "")
    headline = by_device.get(headline_device, {})
    one_pass = list(headline.get("one_pass_order", []))
    warm = list(headline.get("warm_order", []))
    survives = bool(headline.get("ordering_survives", True)) if headline else True
    faster: list[FasterThanBaseline] = []
    if baseline_model and headline_device:
        faster = models_faster_than_baseline(
            profiles, baseline=baseline_model, device=headline_device
        )

    if not profiles:
        verdict = "no encoder profiles recorded"
    else:
        parts = [
            (
                f"{device}: warm {'MATCHES' if info['ordering_survives'] else 'DIFFERS from'} "
                f"one-pass"
            )
            for device, info in by_device.items()
        ]
        if survives:
            verdict = (
                f"on {headline_device or 'host'}, warm rate ordering MATCHES the one-pass "
                "ordering; the architecture-dependent spread is not an artifact of cold load "
                f"or first-kernel compile alone. " + "; ".join(parts)
            )
        else:
            verdict = (
                f"on {headline_device or 'host'}, warm rate ordering DIFFERS from the one-pass "
                "ordering; prefer warm chunks/s for host recommendations "
                f"(one-pass={one_pass}, warm={warm}). " + "; ".join(parts)
            )
        if faster:
            names = ", ".join(
                f"{row['model']} ({row['warm_chunks_per_s']:.1f} c/s, "
                f"{row['speedup_vs_baseline']:.2f}x)"
                if row["speedup_vs_baseline"] is not None
                else f"{row['model']} ({row['warm_chunks_per_s']:.1f} c/s)"
                for row in faster
            )
            verdict += (
                f" Faster than baseline `{baseline_model}` on warm {headline_device}: {names}."
            )
        elif baseline_model:
            verdict += (
                f" No profiled encoder beat baseline `{baseline_model}` on warm "
                f"{headline_device} chunks/s."
            )
    return {
        "corpus_n_texts": corpus_n_texts,
        "devices": devices,
        "profiles": list(profiles),
        "by_device": by_device,
        "one_pass_order": one_pass,
        "warm_order": warm,
        "ordering_survives": survives,
        "verdict": verdict,
        "baseline_model": baseline_model,
        "faster_than_baseline": faster,
    }


def format_host_summary(summary: HostThroughputSummary) -> str:
    """ASCII host summary for the encoder-throughput artifact."""
    headline = (
        "cuda"
        if "cuda" in summary["devices"]
        else (summary["devices"][0] if summary["devices"] else "host")
    )
    lines = [
        "[encoder-throughput] decomposed embedder rates",
        f"  texts={summary['corpus_n_texts']} devices={','.join(summary['devices']) or '-'}",
        (
            f"  {'model':<48} {'dev':<5} {'load_s':>7} {'first_s':>7} {'compile_s':>9} "
            f"{'warm_med':>8} {'iqr':>6} {'prec':>6} {'warm_c/s':>8} {'1pass_c/s':>9} stop"
        ),
    ]
    for profile in sorted(summary["profiles"], key=lambda p: (-p["warm_chunks_per_s"], p["model"])):
        lines.append(
            f"  {profile['model']:<48} {profile['device']:<5} "
            f"{profile['load_seconds']:7.2f} {profile['first_pass_seconds']:7.2f} "
            f"{profile['compile_estimate_seconds']:9.2f} "
            f"{profile['warm_median_seconds']:8.2f} {profile['warm_iqr_seconds']:6.2f} "
            f"{profile['warm_relative_precision']:6.3f} "
            f"{profile['warm_chunks_per_s']:8.1f} {profile['one_pass_chunks_per_s']:9.1f} "
            f"{profile['stopping_reason']}"
        )
    lines.append(f"  one-pass order ({headline}): {' > '.join(summary['one_pass_order']) or '-'}")
    lines.append(f"  warm order ({headline}):     {' > '.join(summary['warm_order']) or '-'}")
    lines.append(f"  ordering_survives={summary['ordering_survives']}")
    for device, info in sorted(summary.get("by_device", {}).items()):
        lines.append(
            f"  [{device}] survives={info['ordering_survives']} "
            f"warm={' > '.join(info['warm_order']) or '-'}"
        )
    baseline = summary.get("baseline_model")
    faster = summary.get("faster_than_baseline") or []
    if baseline:
        if faster:
            named = ", ".join(
                f"{row['model']}={row['warm_chunks_per_s']:.1f}c/s"
                f"({row['speedup_vs_baseline']:.2f}x)"
                if row.get("speedup_vs_baseline") is not None
                else f"{row['model']}={row['warm_chunks_per_s']:.1f}c/s"
                for row in faster
            )
            lines.append(f"  faster_than_baseline ({baseline}): {named}")
        else:
            lines.append(f"  faster_than_baseline ({baseline}): (none)")
    lines.append(f"  verdict: {summary['verdict']}")
    return "\n".join(lines)


def render_host_markdown(summary: HostThroughputSummary) -> str:
    """Durable markdown for `$DATA_DIR/encoder-throughput/<run>/report.md`."""
    lines = [
        "# Encoder throughput decomposition",
        "",
        f"- texts: {summary['corpus_n_texts']}",
        f"- devices: {', '.join(summary['devices']) or '-'}",
        f"- ordering survives warm measurement: `{summary['ordering_survives']}`",
        f"- baseline: `{summary.get('baseline_model') or '-'}`",
        f"- verdict: {summary['verdict']}",
        "",
        "| model | device | load_s | first_s | compile_est_s | warm_median_s | warm_iqr_s "
        "| rel_precision | warm_chunks/s | one_pass_chunks/s | warm_passes | stop | "
        "peak_vram_mb | mean_power_w | power_limit_w |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | "
        "---: | ---: | ---: |",
    ]
    for profile in sorted(summary["profiles"], key=lambda p: (-p["warm_chunks_per_s"], p["model"])):
        lines.append(
            f"| `{profile['model']}` | {profile['device']} | {profile['load_seconds']:.3f} | "
            f"{profile['first_pass_seconds']:.3f} | {profile['compile_estimate_seconds']:.3f} | "
            f"{profile['warm_median_seconds']:.3f} | {profile['warm_iqr_seconds']:.3f} | "
            f"{profile['warm_relative_precision']:.3f} | {profile['warm_chunks_per_s']:.1f} | "
            f"{profile['one_pass_chunks_per_s']:.1f} | {profile['warm_passes']} | "
            f"{profile['stopping_reason']} | {profile.get('peak_vram_mb', '-')} | "
            f"{profile.get('mean_power_w', '-')} | {profile.get('power_limit_w', '-')} |"
        )
    faster = summary.get("faster_than_baseline") or []
    if summary.get("baseline_model"):
        if faster:
            named = ", ".join(
                f"`{row['model']}` ({row['warm_chunks_per_s']:.1f} c/s, "
                f"{row['speedup_vs_baseline']:.2f}x)"
                if row.get("speedup_vs_baseline") is not None
                else f"`{row['model']}` ({row['warm_chunks_per_s']:.1f} c/s)"
                for row in faster
            )
            faster_line = f"- faster than baseline on warm headline device: {named}"
        else:
            faster_line = "- faster than baseline on warm headline device: (none)"
    else:
        faster_line = "- faster than baseline on warm headline device: (baseline not set)"
    lines += [
        "",
        f"- one-pass order: {' > '.join(f'`{m}`' for m in summary['one_pass_order']) or '-'}",
        f"- warm order: {' > '.join(f'`{m}`' for m in summary['warm_order']) or '-'}",
        faster_line,
        "",
    ]
    return "\n".join(lines)


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
