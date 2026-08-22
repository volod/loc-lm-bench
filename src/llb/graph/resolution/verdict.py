"""What the paired rerun decided, per threshold and overall.

The rule the task was written with: a threshold that lifts no lane metric is a NEGATIVE result and
the overlay is not adopted. So a recommendation exists only where a paired reading separated an
overlay lane from the lane it would replace, on the comparison's own bars -- and "recommended" is
still one step short of applied, because no overlay is written into a shipped store here.
"""

from collections.abc import Sequence

from llb.core.contracts.common import JsonObject
from llb.graph.resolution.compare import base_lane, overlay_lane
from llb.graph.resolution.overlay import NodeOverlay
from llb.rag.comparison.models import ComparisonReport
from llb.rag.comparison.uncertainty import DECISION_ADOPT

DECISION_RECOMMEND = "recommend-overlay"
DECISION_NEGATIVE = "negative-result"

# The lane metrics the reading is taken on: the retrieval axis the graph and vector lanes share.
LANE_METRICS = ("recall_at_k", "mrr")


def lane_metrics(report: ComparisonReport, lane: str) -> JsonObject:
    """One lane's scored metrics, or an empty map when the lane was not scored."""
    row = report["backends"].get(lane)
    if row is None:
        return {}
    return {metric: row[metric] for metric in LANE_METRICS if metric in row}  # type: ignore[literal-required]


def threshold_rows(
    reports: dict[str, ComparisonReport], overlays: Sequence[NodeOverlay]
) -> list[JsonObject]:
    """One row per candidate cut: what it merged, and what each strategy scored under it."""
    rows: list[JsonObject] = []
    for overlay in overlays:
        strategies: JsonObject = {}
        for strategy, report in reports.items():
            lane = overlay_lane(strategy, overlay.threshold)
            baseline = lane_metrics(report, base_lane(strategy))
            scored = lane_metrics(report, lane)
            strategies[strategy] = {
                "lane": lane,
                **scored,
                **{
                    f"delta_{metric}": scored[metric] - baseline[metric]
                    for metric in LANE_METRICS
                    if metric in scored and metric in baseline
                },
            }
        rows.append({**overlay.summary(), "strategies": strategies})
    return rows


def baseline_rows(reports: dict[str, ComparisonReport]) -> JsonObject:
    """What each strategy scored BEFORE any overlay -- the reading the deltas are against."""
    return {
        strategy: {"lane": base_lane(strategy), **lane_metrics(report, base_lane(strategy))}
        for strategy, report in reports.items()
    }


def decide(reports: dict[str, ComparisonReport]) -> JsonObject:
    """Recommend the separated overlay lane, or record the negative result and say why.

    A strategy whose own verdict adopted its baseline is not evidence against the overlay in some
    other strategy, so the scan is an OR across strategies and the reason names every one of them.
    """
    adopted = [
        (strategy, report["verdict"])
        for strategy, report in sorted(reports.items())
        if report["verdict"]["decision"] == DECISION_ADOPT
        and report["verdict"]["lane"] != base_lane(strategy)
    ]
    if adopted:
        strategy, verdict = adopted[0]
        return {
            "decision": DECISION_RECOMMEND,
            "strategy": strategy,
            "lane": verdict["lane"],
            "reason": verdict["reason"],
            "note": (
                "a recommendation, not an application: the overlay is written beside the graph "
                "and no shipped store is rewritten by this run"
            ),
        }
    return {
        "decision": DECISION_NEGATIVE,
        "strategy": None,
        "lane": None,
        "reason": "no candidate threshold separated from its pre-overlay lane; "
        + "; ".join(
            f"{strategy}: {report['verdict']['reason']}"
            for strategy, report in sorted(reports.items())
        ),
        "note": "the overlay is not adopted, and the negative reading is the run's result",
    }
