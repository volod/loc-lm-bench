"""Question-routing safety section for answer-quality reports.

A routed row applies the fusion knobs PER QUESTION TYPE, so it is bought for one reason: the slices
the fixed row pays on are the ones it never fuses. Whether it delivers that is a per-slice question
with a per-model answer -- one generator can pay on `factoid` (which routing sends to the baseline
lane) and another on the focus slice itself (which routing deliberately sends TO fusion), and a
section that only checked `factoid` would report the second case as a clean pass.

This module therefore reads three things off the routed lane and names them: the focus-slice
coverage the route keeps, every slice it reproduces the baseline on EXACTLY, and every slice whose
objective it still lowers by an interval clear of zero -- each of those marked with whether it
clears the minimum-evidence gate, since a loss resting on three differing items is not a loss. When
the routed row's fixed twin is in the same comparison, the costs are also read against it: which of
the twin's costs the route cleared, which it kept, and which are new.
"""

from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    AnswerQualityReport,
    LaneReport,
)
from llb.eval.answer_quality.verdict import objective_costs
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs
from llb.rag.fusion_evidence.models import FUSED_ROW_PREFIX, ROUTED_ROW_PREFIX
from llb.rag.fusion_evidence.paired import (
    PairedComparison,
    compared_pairs,
    discordant_pairs,
)
from llb.rag.fusion_evidence.stats import format_interval


def fixed_twin_label(routed_label: str) -> str:
    """The fixed-weight row carrying the same knobs -- what a routed row is bought against."""
    return FUSED_ROW_PREFIX + routed_label[len(ROUTED_ROW_PREFIX) :]


def passthrough_slices(lane: LaneReport) -> list[str]:
    """Slices this lane reproduces the baseline on item for item -- no win, no loss.

    The routing safety claim is exactness, not smallness: a slice the router sends to the baseline
    lane retrieves the same context and generates the same answer, so anything short of a full tie
    means the route did fuse there.
    """
    return sorted(
        name
        for name, entry in lane["slices"].items()
        if entry["n"] > 0 and discordant_pairs(entry["paired_vs_baseline"][METRIC_OBJECTIVE]) == 0
    )


def _cost_line(name: str, paired: PairedComparison, gated: bool, confidence: float) -> str:
    ledger = (
        f"{paired['wins']}/{paired['losses']}/{paired['ties']} w/l/t, "
        f"{discordant_pairs(paired)} of {compared_pairs(paired)} differing"
    )
    verdict = (
        "clears the minimum-evidence gate"
        if gated
        else (
            f"does NOT clear the minimum-evidence gate ({minimum_discordant_pairs(confidence)} "
            "differing items needed), so it is an open question rather than a measured loss"
        )
    )
    return (
        f"  - still pays on `{name}`: objective {format_interval(paired['delta'])} "
        f"({ledger}) -- {verdict}"
    )


def _twin_line(routed: LaneReport, twin: LaneReport | None, confidence: float) -> list[str]:
    """What the route cleared, kept, and added relative to the fixed row carrying its knobs."""
    if twin is None:
        return []
    fixed = {name for name, _paired, gated in objective_costs(twin, confidence) if gated}
    routed_costs = {name for name, _paired, gated in objective_costs(routed, confidence) if gated}
    parts = [
        f"clears {_names(fixed - routed_costs)}",
        f"retains {_names(fixed & routed_costs)}",
        f"adds {_names(routed_costs - fixed)}",
    ]
    return [f"  - against its fixed twin `{twin['label']}`: " + "; ".join(parts)]


def _names(names: set[str]) -> str:
    return ", ".join(f"`{name}`" for name in sorted(names)) if names else "nothing"


def _lane_lines(
    report: AnswerQualityReport, label: str, lane: LaneReport, coverage: str
) -> list[str]:
    confidence = report["confidence"]
    focus = lane["slices"].get(report["focus_slice"])
    if focus is None:
        return [f"- `{label}`: the focus slice is absent; no routing claim."]
    focus_delta = focus["paired_vs_baseline"][coverage]["delta"]
    passthrough = passthrough_slices(lane)
    costs = objective_costs(lane, confidence)
    lines = [
        f"- `{label}`: {coverage} {format_interval(focus_delta)} on "
        f"{report['focus_slice']}; exact baseline passthrough on {_names(set(passthrough))}."
    ]
    if not costs:
        lines.append(
            "  - no slice's objective falls by an interval clear of zero: the route leaves "
            "nothing measurably paying."
        )
    lines += [_cost_line(name, paired, gated, confidence) for name, paired, gated in costs]
    twin = report["lanes"].get(fixed_twin_label(label))
    lines += _twin_line(lane, twin, confidence)
    return lines


def routing_outcomes(report: AnswerQualityReport) -> list[str]:
    """Render what each routed lane keeps, passes through, and still pays for."""
    routed = {
        label: lane
        for label, lane in report["lanes"].items()
        if label.startswith(ROUTED_ROW_PREFIX)
    }
    if not routed:
        return []
    coverage = report["verdict"]["coverage_metric"]
    lines = [
        "### Routing outcome",
        "",
        "A routed row fuses only the question types that need linked evidence, so it is bought to "
        "clear the slices its fixed twin pays on. What it keeps, what it reproduces exactly, and "
        "what it STILL pays on are per-slice readings, and the last one is model-dependent: a "
        "cost landing on the focus slice is a cost routing cannot clear by construction.",
        "",
    ]
    for label in sorted(routed):
        lines += _lane_lines(report, label, routed[label], coverage)
    lines.append("")
    return lines


__all__ = ["fixed_twin_label", "passthrough_slices", "routing_outcomes"]
