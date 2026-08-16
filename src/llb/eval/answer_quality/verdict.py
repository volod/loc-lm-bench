"""Turn the scored lanes into one sentence about whether retrieval coverage reached the answer.

The gate is on the OBJECTIVE, not on retrieval: a fused lane that retrieves more evidence but
answers no better has produced a retrieval-only effect, and saying so is the point of this lane.
As in the fusion-evidence verdict, the decision reads the calibrated paired sign-flip p rather than
the point estimate or bootstrap interval.

Order matters. `retrieval_only` is checked BEFORE `inconclusive`, because a coverage gain whose own
calibrated test separates is a MEASURED result about the retrieval half; calling that case
`inconclusive` on the strength of a +0.011 objective would report the noisy half and drop the
measured one.
"""

from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    LaneDecision,
    LaneReport,
    AnswerQualityVerdict,
    VERDICT_ANSWER_GAIN,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_NO_GAIN,
    VERDICT_RETRIEVAL_ONLY,
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

ZERO: Interval = {"mean": 0.0, "lo": 0.0, "hi": 0.0}
NO_COMPARISON: PairedComparison = {
    "delta": ZERO,
    "wins": 0,
    "losses": 0,
    "ties": 0,
    "sign_test_p": 1.0,
}


def _focus_paired(lane: LaneReport, focus_slice: str, metric: str) -> PairedComparison:
    slice_report = lane["slices"].get(focus_slice)
    if slice_report is None:
        return NO_COMPARISON
    return slice_report["paired_vs_baseline"][metric]


def _focus_delta(lane: LaneReport, focus_slice: str, metric: str) -> Interval:
    return _focus_paired(lane, focus_slice, metric)["delta"]


def _focus_stability(lane: LaneReport, focus_slice: str, metric: str) -> ReadingStability | None:
    """How settled one focus-slice reading is, or None on an artifact carrying no draw."""
    slice_report = lane["slices"].get(focus_slice)
    if slice_report is None:
        return None
    return slice_report["paired_vs_baseline"][metric].get("stability")


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
    confidence: float = DEFAULT_CONFIDENCE,
) -> AnswerQualityVerdict:
    """Judge every candidate lane on the focus slice and name the strongest one.

    `coverage` names the retrieval metric a retrieval-only effect is stated on -- `all_spans_at_k`
    when the run bundles carried the retrieval sidecar, the weaker any-span hit otherwise. Every
    candidate keeps its own decision in `lane_decisions`: a three-lane comparison has a result per
    lane, and collapsing it to the winner would silently drop the others.
    """
    verdict: AnswerQualityVerdict = {
        "focus_slice": focus_slice,
        "focus_n": _focus_n(lanes, baseline, focus_slice),
        "baseline": baseline,
        "best_lane": None,
        "coverage_metric": coverage,
        "decision": VERDICT_NO_EVIDENCE,
        "reason": "",
        "lane_decisions": {},
    }
    candidates = {label: lane for label, lane in lanes.items() if label != baseline}
    if not candidates:
        verdict["reason"] = "no lane was compared against the baseline"
        return verdict
    if verdict["focus_n"] == 0:
        verdict["reason"] = f"the scored set has no {focus_slice} item"
        return verdict
    decisions: dict[str, LaneDecision] = {}
    for label in sorted(candidates):
        decision, reason = judge_lane(
            candidates[label], label, baseline, focus_slice, coverage, confidence
        )
        decisions[label] = {"decision": decision, "reason": reason}
    best = max(sorted(candidates), key=lambda label: _rank_key(candidates[label], focus_slice))
    verdict["best_lane"] = best
    verdict["decision"] = decisions[best]["decision"]
    verdict["reason"] = decisions[best]["reason"]
    verdict["lane_decisions"] = decisions
    return verdict


def _focus_n(lanes: dict[str, LaneReport], baseline: str, focus_slice: str) -> int:
    lane = lanes.get(baseline)
    if lane is None:
        return 0
    slice_report = lane["slices"].get(focus_slice)
    return slice_report["n"] if slice_report else 0


def judge_lane(
    lane: LaneReport,
    label: str,
    baseline: str,
    focus_slice: str,
    coverage_metric: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[str, str]:
    """The `(decision, reason)` for one candidate lane against the lane it is read against.

    Public because a budget sweep reads the same row at two budgets against each other rather than
    against the report baseline, and that comparison must reach the SAME four outcomes with the
    same calibrated test -- a second verdict vocabulary would make the artifact unreadable.
    """
    paired_objective = _focus_paired(lane, focus_slice, METRIC_OBJECTIVE)
    paired_coverage = _focus_paired(lane, focus_slice, coverage_metric)
    objective = paired_objective["delta"]
    coverage = paired_coverage["delta"]
    detail = (
        f"objective {objective['mean']:+.3f} "
        f"[{objective['lo']:+.3f}, {objective['hi']:+.3f}], "
        f"{coverage_metric} {coverage['mean']:+.3f} "
        f"[{coverage['lo']:+.3f}, {coverage['hi']:+.3f}]"
    )
    # Both readings the four outcomes below are cut from, so whichever branch fires says when the
    # cut -- not the evidence -- is what produced it.
    note = borderline_note(
        [
            (metric, _focus_stability(lane, focus_slice, metric))
            for metric in (METRIC_OBJECTIVE, coverage_metric)
        ]
    ) + evidence_gate_clause(
        [(METRIC_OBJECTIVE, paired_objective), (coverage_metric, paired_coverage)], confidence
    )
    if separates(paired_objective, confidence):
        return VERDICT_ANSWER_GAIN, (
            f"{label} answers {focus_slice} better than {baseline} ({detail}); the retrieval gain "
            "reaches the answer" + note
        )
    # A coverage gain whose calibrated test separates while the objective's does not IS
    # the retrieval-only finding -- reporting it as merely `inconclusive` would throw away the
    # measured half of the result.
    if separates(paired_coverage, confidence):
        return VERDICT_RETRIEVAL_ONLY, (
            f"{label} carries {coverage['mean']:+.3f} more of the {focus_slice} evidence than "
            f"{baseline}, but its objective is not separable from it ({detail}); the coverage gain "
            "is a retrieval-only effect" + note
        )
    if objective["mean"] > 0.0:
        return VERDICT_INCONCLUSIVE, (
            f"{label} gains {objective['mean']:+.3f} objective on {focus_slice} but the calibrated "
            f"test does not separate ({detail}); a larger {focus_slice} slice is needed to "
            "separate it from " + baseline + note
        )
    return VERDICT_NO_GAIN, (
        f"{label} neither retrieves nor answers {focus_slice} better than {baseline} ({detail})"
        + note
    )
