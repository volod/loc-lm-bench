"""Turn the scored rows into one adopt-or-reject sentence about graph-vector fusion.

The gate is deliberately asymmetric. Fusion is an OPT-IN backend, so it has to earn a default by
recovering multi-hop evidence the vector lane misses WITHOUT paying for it in overall recall. A tie
everywhere is a reject, not an adopt: it would add a graph build and a second retrieval lane for
nothing.

The gate is on the INTERVAL, not the point estimate. A multi-hop slice is a dozen or a few dozen
items, so a `+0.086` mean recall gain whose paired bootstrap interval is `[0.000, 0.200]` is a
plausible gain and no more -- calling that an adopt would waste the uncertainty this lane exists to
produce. Such a row is `inconclusive`: the direction is recorded, the recommendation is not.
"""

from llb.rag.fusion_evidence.models import (
    FUSED_ROW_PREFIX,
    METRIC_ALL_SPANS,
    METRIC_RECALL,
    OVERALL_RECALL_TOLERANCE,
    RowReport,
    ROUTED_ROW_PREFIX,
    Verdict,
    VERDICT_ADOPT,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_REJECT,
)
from llb.rag.fusion_evidence.stats import Interval

# The two focus-slice metrics a fused row may earn its default on: recovering ANY hop the vector
# lane missed, or completing the evidence of an item it only half-covered.
GAIN_METRICS = (METRIC_RECALL, METRIC_ALL_SPANS)


def _focus_delta(row: RowReport, focus_slice: str, metric: str) -> Interval:
    slice_report = row["slices"].get(focus_slice)
    if slice_report is None:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    return slice_report["paired_vs_baseline"][metric]["delta"]


def _overall_delta(row: RowReport, metric: str) -> Interval:
    return row["overall"]["paired_vs_baseline"][metric]["delta"]


def _rank_key(row: RowReport, focus_slice: str) -> tuple[float, float, float, float]:
    """Rank by evidence strength first: a gain whose interval clears zero beats a larger mean."""
    return (
        max(_focus_delta(row, focus_slice, metric)["lo"] for metric in GAIN_METRICS),
        _focus_delta(row, focus_slice, METRIC_RECALL)["mean"],
        _focus_delta(row, focus_slice, METRIC_ALL_SPANS)["mean"],
        _overall_delta(row, METRIC_RECALL)["mean"],
    )


def decide(rows: dict[str, RowReport], *, baseline: str, focus_slice: str) -> Verdict:
    """Pick the best fused row on the focus slice and state whether it earns a default."""
    focus_n = _focus_n(rows, baseline, focus_slice)
    verdict: Verdict = {
        "focus_slice": focus_slice,
        "focus_n": focus_n,
        "baseline": baseline,
        "best_row": None,
        "decision": VERDICT_NO_EVIDENCE,
        "reason": "",
    }
    candidates = {
        label: row
        for label, row in rows.items()
        if label.startswith((FUSED_ROW_PREFIX, ROUTED_ROW_PREFIX)) and label != baseline
    }
    if not candidates:
        verdict["reason"] = "no fused row was compared"
        return verdict
    if focus_n == 0:
        verdict["reason"] = f"the scored set has no {focus_slice} item"
        return verdict
    best = max(sorted(candidates), key=lambda label: _rank_key(candidates[label], focus_slice))
    decision, reason = _judge(candidates[best], best, focus_slice)
    verdict["best_row"] = best
    verdict["decision"] = decision
    verdict["reason"] = reason
    return verdict


def _focus_n(rows: dict[str, RowReport], baseline: str, focus_slice: str) -> int:
    row = rows.get(baseline)
    if row is None:
        return 0
    slice_report = row["slices"].get(focus_slice)
    return slice_report["n"] if slice_report else 0


def _judge(row: RowReport, label: str, focus_slice: str) -> tuple[str, str]:
    """The `(decision, reason)` for the winning fused row."""
    gains = {metric: _focus_delta(row, focus_slice, metric) for metric in GAIN_METRICS}
    overall = _overall_delta(row, METRIC_RECALL)
    best_mean = max(gain["mean"] for gain in gains.values())
    best_lo = max(gain["lo"] for gain in gains.values())
    detail = (
        f"recall {gains[METRIC_RECALL]['mean']:+.3f} "
        f"[{gains[METRIC_RECALL]['lo']:+.3f}, {gains[METRIC_RECALL]['hi']:+.3f}], "
        f"all-spans {gains[METRIC_ALL_SPANS]['mean']:+.3f} "
        f"[{gains[METRIC_ALL_SPANS]['lo']:+.3f}, {gains[METRIC_ALL_SPANS]['hi']:+.3f}]"
    )
    if best_mean <= 0.0:
        return VERDICT_REJECT, (
            f"{label} does not beat the vector lane on {focus_slice} ({detail}); "
            "fusion stays opt-in"
        )
    if best_lo <= 0.0:
        return VERDICT_INCONCLUSIVE, (
            f"{label} gains {best_mean:+.3f} on {focus_slice} but the interval includes no "
            f"difference ({detail}); fusion stays opt-in until a larger {focus_slice} slice "
            "separates it from the vector lane"
        )
    if overall["mean"] < -OVERALL_RECALL_TOLERANCE:
        return VERDICT_REJECT, (
            f"{label} gains {best_mean:+.3f} on {focus_slice} but costs "
            f"{overall['mean']:+.3f} overall recall@k; fusion stays opt-in"
        )
    return VERDICT_ADOPT, (
        f"{label} gains {best_mean:+.3f} on {focus_slice} ({detail}) with "
        f"{overall['mean']:+.3f} [{overall['lo']:+.3f}, {overall['hi']:+.3f}] overall recall@k"
    )
