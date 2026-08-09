"""Turn the scored rows into one adopt-or-reject sentence about graph-vector fusion.

The gate is deliberately asymmetric. Fusion is an OPT-IN backend, so it has to earn a default by
recovering multi-hop evidence the vector lane misses WITHOUT paying for it in overall recall. A tie
everywhere is a reject, not an adopt: it would add a graph build and a second retrieval lane for
nothing.

The gate is on both the calibrated per-row sign-flip p and the selected grid family's adjusted p,
not the point estimate or percentile interval. A positive mean that does not clear both cuts is
plausible gain and no more. Such a row is `inconclusive`: the direction is recorded, the
recommendation is not.
"""

from dataclasses import dataclass

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
from llb.rag.fusion_evidence.stability import (
    ReadingStability,
    borderline_note,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    Interval,
)
from llb.rag.fusion_evidence.paired import (
    PairedComparison,
    evidence_gate_clause,
    separates,
)
from llb.rag.fusion_evidence.selection import (
    SelectionAdjustment,
    selection_separates,
)
from llb.rag.fusion_evidence.selection_family import hypothesis_key

# The two focus-slice metrics a fused row may earn its default on: recovering ANY hop the vector
# lane missed, or completing the evidence of an item it only half-covered.
GAIN_METRICS = (METRIC_RECALL, METRIC_ALL_SPANS)

_ZERO: Interval = {"mean": 0.0, "lo": 0.0, "hi": 0.0}
_NO_COMPARISON: PairedComparison = {
    "delta": _ZERO,
    "wins": 0,
    "losses": 0,
    "ties": 0,
    "sign_test_p": 1.0,
}


def _focus_paired(row: RowReport, focus_slice: str, metric: str) -> PairedComparison:
    slice_report = row["slices"].get(focus_slice)
    if slice_report is None:
        return _NO_COMPARISON
    return slice_report["paired_vs_baseline"][metric]


def _focus_delta(row: RowReport, focus_slice: str, metric: str) -> Interval:
    return _focus_paired(row, focus_slice, metric)["delta"]


def _focus_stability(row: RowReport, focus_slice: str, metric: str) -> ReadingStability | None:
    """How settled the focus-slice reading of one gain metric is, or None on an archived report."""
    slice_report = row["slices"].get(focus_slice)
    if slice_report is None:
        return None
    return slice_report["paired_vs_baseline"][metric].get("stability")


def _gain_note(row: RowReport, focus_slice: str) -> str:
    """The shared borderline clause over the gain metrics this verdict was decided on.

    A `reject` and an `adopt` are both cuts of the same calibrated p, so both need to say when the cut
    -- rather than the evidence -- is what produced them.
    """
    return borderline_note(
        [(metric, _focus_stability(row, focus_slice, metric)) for metric in GAIN_METRICS]
    )


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


def _candidates(rows: dict[str, RowReport]) -> dict[str, RowReport]:
    return {
        label: row
        for label, row in rows.items()
        if label.startswith((FUSED_ROW_PREFIX, ROUTED_ROW_PREFIX))
    }


def decide(
    rows: dict[str, RowReport],
    *,
    baseline: str,
    focus_slice: str,
    confidence: float = DEFAULT_CONFIDENCE,
    adjustment: SelectionAdjustment | None = None,
) -> Verdict:
    """Pick the best fused row on the focus slice and state whether it earns a default.

    The ranking is untouched by the minimum-evidence gate on purpose: which row leads is a summary
    of the grid, and re-ordering it would be an adoption-rule change. What the gate decides is
    whether the leading row's gain may be READ as a separation.
    """
    focus_n = _focus_n(rows, baseline, focus_slice)
    verdict: Verdict = {
        "focus_slice": focus_slice,
        "focus_n": focus_n,
        "baseline": baseline,
        "best_row": None,
        "decision": VERDICT_NO_EVIDENCE,
        "reason": "",
    }
    candidates = {label: row for label, row in _candidates(rows).items() if label != baseline}
    if not candidates:
        verdict["reason"] = "no fused row was compared"
        return verdict
    if focus_n == 0:
        verdict["reason"] = f"the scored set has no {focus_slice} item"
        return verdict
    best = max(sorted(candidates), key=lambda label: _rank_key(candidates[label], focus_slice))
    decision, reason = _judge(candidates[best], best, focus_slice, confidence, adjustment)
    verdict["best_row"] = best
    verdict["decision"] = decision
    verdict["reason"] = reason
    if adjustment is not None:
        verdict["selection_adjustment"] = adjustment
    return verdict


def _focus_n(rows: dict[str, RowReport], baseline: str, focus_slice: str) -> int:
    row = rows.get(baseline)
    if row is None:
        return 0
    slice_report = row["slices"].get(focus_slice)
    return slice_report["n"] if slice_report else 0


@dataclass(frozen=True, slots=True)
class _Reading:
    """One winning row read against the vector lane: the gains, and which of them may be read.

    Assembled before any verdict branch, because the four decisions below all rest on the same four
    facts and reading them per branch is how two branches come to disagree.
    """

    gains: dict[str, Interval]
    overall: Interval
    best_mean: float
    per_row_separated: list[str]
    separated: list[str]
    detail: str
    note: str


def _reading_of(
    row: RowReport,
    label: str,
    focus_slice: str,
    confidence: float,
    adjustment: SelectionAdjustment | None,
) -> _Reading:
    """Pair the focus slice on every gain metric and apply the reading rules to the result."""
    paired = {metric: _focus_paired(row, focus_slice, metric) for metric in GAIN_METRICS}
    gains = {metric: comparison["delta"] for metric, comparison in paired.items()}
    per_row_separated = [
        metric for metric, comparison in paired.items() if separates(comparison, confidence)
    ]
    return _Reading(
        gains=gains,
        overall=_overall_delta(row, METRIC_RECALL),
        best_mean=max(gain["mean"] for gain in gains.values()),
        per_row_separated=per_row_separated,
        separated=[
            metric
            for metric in per_row_separated
            if adjustment is None
            or selection_separates(adjustment, hypothesis_key(label, metric), confidence)
        ],
        detail=(
            f"recall {gains[METRIC_RECALL]['mean']:+.3f} "
            f"[{gains[METRIC_RECALL]['lo']:+.3f}, {gains[METRIC_RECALL]['hi']:+.3f}], "
            f"all-spans {gains[METRIC_ALL_SPANS]['mean']:+.3f} "
            f"[{gains[METRIC_ALL_SPANS]['lo']:+.3f}, {gains[METRIC_ALL_SPANS]['hi']:+.3f}]"
        ),
        note=(
            _gain_note(row, focus_slice)
            + evidence_gate_clause(
                [(metric, paired[metric]) for metric in GAIN_METRICS], confidence
            )
            + _selection_note(label, adjustment)
        ),
    )


def _inconclusive_limit(reading: _Reading) -> str:
    """WHICH rule stopped a positive gain from being read: selection, randomization, or evidence."""
    if reading.per_row_separated:
        return "the family-wise selection adjustment does not separate"
    if all(gain["lo"] <= 0.0 for gain in reading.gains.values()):
        return "the calibrated randomization test does not separate"
    return "no gain clear of zero rests on enough differing items to be read as one"


def _judge(
    row: RowReport,
    label: str,
    focus_slice: str,
    confidence: float = DEFAULT_CONFIDENCE,
    adjustment: SelectionAdjustment | None = None,
) -> tuple[str, str]:
    """The `(decision, reason)` for the winning fused row."""
    reading = _reading_of(row, label, focus_slice, confidence, adjustment)
    if reading.best_mean <= 0.0:
        return VERDICT_REJECT, (
            f"{label} does not beat the vector lane on {focus_slice} ({reading.detail}); "
            "fusion stays opt-in" + reading.note
        )
    if not reading.separated:
        return VERDICT_INCONCLUSIVE, (
            f"{label} gains {reading.best_mean:+.3f} on {focus_slice} but "
            f"{_inconclusive_limit(reading)} ({reading.detail}); fusion stays opt-in until a "
            f"larger {focus_slice} slice separates it from the vector lane" + reading.note
        )
    if reading.overall["mean"] < -OVERALL_RECALL_TOLERANCE:
        return VERDICT_REJECT, (
            f"{label} gains {reading.best_mean:+.3f} on {focus_slice} but costs "
            f"{reading.overall['mean']:+.3f} overall recall@k; fusion stays opt-in" + reading.note
        )
    return VERDICT_ADOPT, (
        f"{label} gains {reading.best_mean:+.3f} on {focus_slice} ({reading.detail}) with "
        f"{reading.overall['mean']:+.3f} [{reading.overall['lo']:+.3f}, "
        f"{reading.overall['hi']:+.3f}] overall recall@k" + reading.note
    )


def _selection_note(label: str, adjustment: SelectionAdjustment | None) -> str:
    if adjustment is None:
        return ""
    readings = [
        (
            metric,
            adjustment["p_values"][hypothesis_key(label, metric)]["unadjusted_p"],
            adjustment["p_values"][hypothesis_key(label, metric)]["adjusted_p"],
        )
        for metric in GAIN_METRICS
    ]
    detail = ", ".join(
        f"{metric} raw p={raw:.4f}, adjusted p={adjusted:.4f}" for metric, raw, adjusted in readings
    )
    return (
        f"; selection adjustment ({adjustment['family_size']} row-metric hypotheses, "
        f"Westfall-Young step-down max-T): {detail}"
    )
