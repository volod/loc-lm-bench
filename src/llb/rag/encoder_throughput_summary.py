"""Read a set of encoder profiles as one host answer: per-device order, survival, verdict.

Separate from the measurement next door because these are two different claims. Measuring one
encoder is arithmetic over one process; ordering several of them -- and saying whether the warm
ordering reproduces the one-pass ordering -- is a reading over the whole host, and it is the reading
a recommendation quotes.
"""

from collections.abc import Sequence
from typing import Any

from llb.rag.encoder_throughput import FasterThanBaseline, HostThroughputSummary, ThroughputProfile


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


def _orders_by_device(profiles: Sequence[ThroughputProfile]) -> dict[str, dict[str, Any]]:
    """Each device's one-pass and warm rate order, and whether the two agree."""
    by_device: dict[str, dict[str, Any]] = {}
    for device in sorted({profile["device"] for profile in profiles}):
        group = [profile for profile in profiles if profile["device"] == device]
        one_pass = rate_order(group, warm=False)
        warm = rate_order(group, warm=True)
        by_device[device] = {
            "one_pass_order": one_pass,
            "warm_order": warm,
            "ordering_survives": one_pass == warm,
        }
    return by_device


def _headline_device(by_device: dict[str, dict[str, Any]], devices: list[str]) -> str:
    """The device the host recommendation is read off: CUDA when present, else what there is."""
    if "cuda" in by_device:
        return "cuda"
    return devices[0] if devices else ""


def _faster_clause(
    faster: list[FasterThanBaseline], baseline_model: str | None, device: str
) -> str:
    """The sentence naming which encoders beat the baseline, or that none did."""
    if not baseline_model:
        return ""
    if not faster:
        return f" No profiled encoder beat baseline `{baseline_model}` on warm {device} chunks/s."
    names = ", ".join(
        f"{row['model']} ({row['warm_chunks_per_s']:.1f} c/s, {row['speedup_vs_baseline']:.2f}x)"
        if row["speedup_vs_baseline"] is not None
        else f"{row['model']} ({row['warm_chunks_per_s']:.1f} c/s)"
        for row in faster
    )
    return f" Faster than baseline `{baseline_model}` on warm {device}: {names}."


def _ordering_verdict(
    by_device: dict[str, dict[str, Any]],
    *,
    device: str,
    survives: bool,
    one_pass: list[str],
    warm: list[str],
) -> str:
    """Whether warm ordering reproduces the one-pass ordering, which decides what may be quoted."""
    parts = "; ".join(
        f"{name}: warm {'MATCHES' if info['ordering_survives'] else 'DIFFERS from'} one-pass"
        for name, info in by_device.items()
    )
    if survives:
        return (
            f"on {device or 'host'}, warm rate ordering MATCHES the one-pass ordering; the "
            "architecture-dependent spread is not an artifact of cold load or first-kernel "
            f"compile alone. {parts}"
        )
    return (
        f"on {device or 'host'}, warm rate ordering DIFFERS from the one-pass ordering; prefer "
        f"warm chunks/s for host recommendations (one-pass={one_pass}, warm={warm}). {parts}"
    )


def build_host_summary(
    profiles: Sequence[ThroughputProfile],
    *,
    corpus_n_texts: int,
    baseline_model: str | None = None,
) -> HostThroughputSummary:
    """Cross-encoder summary: per-device orders, survival flag, and a one-line verdict."""
    devices = sorted({profile["device"] for profile in profiles})
    by_device = _orders_by_device(profiles)
    device = _headline_device(by_device, devices)
    headline = by_device.get(device, {})
    one_pass = list(headline.get("one_pass_order", []))
    warm = list(headline.get("warm_order", []))
    survives = bool(headline.get("ordering_survives", True)) if headline else True
    faster: list[FasterThanBaseline] = (
        models_faster_than_baseline(profiles, baseline=baseline_model, device=device)
        if baseline_model and device
        else []
    )
    if not profiles:
        verdict = "no encoder profiles recorded"
    else:
        verdict = _ordering_verdict(
            by_device, device=device, survives=survives, one_pass=one_pass, warm=warm
        ) + _faster_clause(faster, baseline_model, device)
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
