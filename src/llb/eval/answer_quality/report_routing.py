"""Question-routing safety section for answer-quality reports."""

from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    AnswerQualityReport,
)
from llb.rag.fusion_evidence.models import ROUTED_ROW_PREFIX
from llb.rag.fusion_evidence.stats import format_interval


def routing_outcomes(report: AnswerQualityReport) -> list[str]:
    """Render focus gain and factoid passthrough for routed lanes."""
    routed = {
        label: lane
        for label, lane in report["lanes"].items()
        if label.startswith(ROUTED_ROW_PREFIX)
    }
    if not routed:
        return []
    coverage = report["verdict"]["coverage_metric"]
    lines = ["### Routing outcome", ""]
    for label in sorted(routed):
        lane = routed[label]
        focus = lane["slices"].get(report["focus_slice"])
        factoid = lane["slices"].get("factoid")
        if focus is None or factoid is None:
            lines.append(f"- `{label}`: the focus or factoid slice is absent; no routing claim.")
            continue
        focus_delta = focus["paired_vs_baseline"][coverage]["delta"]
        factoid_pair = factoid["paired_vs_baseline"][METRIC_OBJECTIVE]
        factoid_delta = factoid_pair["delta"]
        exact_factoid = (
            factoid_delta["mean"] == 0.0
            and factoid_delta["lo"] == 0.0
            and factoid_delta["hi"] == 0.0
        )
        conclusion = (
            "keeps the measured focus-slice coverage gain and makes factoid answers an exact "
            "baseline passthrough"
            if focus_delta["mean"] > 0.0 and exact_factoid
            else "does not establish both a focus-slice coverage gain and exact factoid passthrough"
        )
        lines.append(
            f"- `{label}`: {coverage} {format_interval(focus_delta)} on "
            f"{report['focus_slice']}; factoid objective {format_interval(factoid_delta)} "
            f"({factoid_pair['wins']}/{factoid_pair['losses']}/{factoid_pair['ties']} w/l/t); "
            f"{conclusion}."
        )
    lines.append("")
    return lines
