"""Turn the scored lanes into one sentence about whether retrieval coverage reached the answer.

The gate is on the OBJECTIVE, not on retrieval: a fused lane that retrieves more evidence but
answers no better has produced a retrieval-only effect, and saying so is the point of this lane.
As in the fusion-evidence verdict, the decision reads the paired INTERVAL rather than the point
estimate -- a multi-hop slice is a few dozen items, so a positive mean whose interval includes no
difference is never an adopt.

Order matters. `retrieval_only` is checked BEFORE `inconclusive`, because a coverage gain whose own
interval clears zero is a MEASURED result about the retrieval half; calling that case
`inconclusive` on the strength of a +0.011 objective would report the noisy half and drop the
measured one.
"""

from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    LaneReport,
    AnswerQualityVerdict,
    VERDICT_ANSWER_GAIN,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_NO_GAIN,
    VERDICT_RETRIEVAL_ONLY,
)
from llb.rag.fusion_evidence.stats import Interval

ZERO: Interval = {"mean": 0.0, "lo": 0.0, "hi": 0.0}


def _focus_delta(lane: LaneReport, focus_slice: str, metric: str) -> Interval:
    slice_report = lane["slices"].get(focus_slice)
    if slice_report is None:
        return ZERO
    return slice_report["paired_vs_baseline"][metric]["delta"]


def _rank_key(lane: LaneReport, focus_slice: str) -> tuple[float, float, float]:
    """Rank by evidence strength first: an objective gain clear of zero beats a larger mean."""
    objective = _focus_delta(lane, focus_slice, METRIC_OBJECTIVE)
    overall = lane["overall"]["paired_vs_baseline"][METRIC_OBJECTIVE]["delta"]
    return (objective["lo"], objective["mean"], overall["mean"])


def decide(
    lanes: dict[str, LaneReport],
    *,
    baseline: str,
    focus_slice: str,
    coverage: str = METRIC_RETRIEVAL_HIT,
) -> AnswerQualityVerdict:
    """Pick the best candidate lane on the focus slice and state what its gain amounts to.

    `coverage` names the retrieval metric a retrieval-only effect is stated on -- `all_spans_at_k`
    when the run bundles carried the retrieval sidecar, the weaker any-span hit otherwise.
    """
    verdict: AnswerQualityVerdict = {
        "focus_slice": focus_slice,
        "focus_n": _focus_n(lanes, baseline, focus_slice),
        "baseline": baseline,
        "best_lane": None,
        "coverage_metric": coverage,
        "decision": VERDICT_NO_EVIDENCE,
        "reason": "",
    }
    candidates = {label: lane for label, lane in lanes.items() if label != baseline}
    if not candidates:
        verdict["reason"] = "no lane was compared against the baseline"
        return verdict
    if verdict["focus_n"] == 0:
        verdict["reason"] = f"the scored set has no {focus_slice} item"
        return verdict
    best = max(sorted(candidates), key=lambda label: _rank_key(candidates[label], focus_slice))
    decision, reason = _judge(candidates[best], best, baseline, focus_slice, coverage)
    verdict["best_lane"] = best
    verdict["decision"] = decision
    verdict["reason"] = reason
    return verdict


def _focus_n(lanes: dict[str, LaneReport], baseline: str, focus_slice: str) -> int:
    lane = lanes.get(baseline)
    if lane is None:
        return 0
    slice_report = lane["slices"].get(focus_slice)
    return slice_report["n"] if slice_report else 0


def _judge(
    lane: LaneReport, label: str, baseline: str, focus_slice: str, coverage_metric: str
) -> tuple[str, str]:
    """The `(decision, reason)` for the winning candidate lane."""
    objective = _focus_delta(lane, focus_slice, METRIC_OBJECTIVE)
    coverage = _focus_delta(lane, focus_slice, coverage_metric)
    detail = (
        f"objective {objective['mean']:+.3f} "
        f"[{objective['lo']:+.3f}, {objective['hi']:+.3f}], "
        f"{coverage_metric} {coverage['mean']:+.3f} "
        f"[{coverage['lo']:+.3f}, {coverage['hi']:+.3f}]"
    )
    if objective["lo"] > 0.0:
        return VERDICT_ANSWER_GAIN, (
            f"{label} answers {focus_slice} better than {baseline} ({detail}); the retrieval gain "
            "reaches the answer"
        )
    # A coverage gain whose own interval clears zero, paired with an objective that does not, IS
    # the retrieval-only finding -- reporting it as merely `inconclusive` would throw away the
    # measured half of the result.
    if coverage["lo"] > 0.0:
        return VERDICT_RETRIEVAL_ONLY, (
            f"{label} carries {coverage['mean']:+.3f} more of the {focus_slice} evidence than "
            f"{baseline}, but its objective is not separable from it ({detail}); the coverage gain "
            "is a retrieval-only effect"
        )
    if objective["mean"] > 0.0:
        return VERDICT_INCONCLUSIVE, (
            f"{label} gains {objective['mean']:+.3f} objective on {focus_slice} but the interval "
            f"includes no difference ({detail}); a larger {focus_slice} slice is needed to "
            "separate it from " + baseline
        )
    return VERDICT_NO_GAIN, (
        f"{label} neither retrieves nor answers {focus_slice} better than {baseline} ({detail})"
    )
