"""ASCII Markdown report for tuning-only selection and frozen-final routing evidence."""

from llb.rag.fusion_calibration.models import PolicyResult, RoutingCalibrationReport
from llb.rag.fusion_evidence.evidence_gate import (
    evidence_gate_summary,
)
from llb.rag.fusion_evidence.stats import discordant_pairs, format_interval, gated_readings


def _row(label: str, result: PolicyResult) -> str:
    route = result["route"]
    multi = result["multi_span_coverage"]
    single = result["single_span_recall"]
    gate = "yes" if result["recommendation_gate"] else "no"
    return (
        f"| {label} | {result['graph_questions']}/{result['vector_questions']} "
        f"| {route['true_positive']}/{route['false_positive']}/"
        f"{route['true_negative']}/{route['false_negative']} "
        f"| {format_interval(route['precision'])} | {format_interval(route['recall'])} "
        f"| {format_interval(multi['delta'])} | {discordant_pairs(multi)} "
        f"| {format_interval(single['delta'])} | {gate} |"
    )


def _table(results: dict[str, PolicyResult]) -> list[str]:
    lines = [
        "| policy | graph/vector | tp/fp/tn/fn | precision | recall | multi coverage delta "
        "| d | single recall delta | gate |",
        "| --- | ---: | :-: | ---: | ---: | ---: | ---: | ---: | :-: |",
    ]
    lines.extend(_row(label, results[label]) for label in sorted(results))
    lines.append("")
    return lines


def _gate_summary(report: RoutingCalibrationReport) -> list[str]:
    """How many multi-span coverage readings the minimum-evidence gate relabeled."""
    results = [*report["tuning"].values(), report["final"]]
    gated, total = gated_readings(
        [result["multi_span_coverage"] for result in results], report["confidence"]
    )
    return evidence_gate_summary(
        gated, total, report["confidence"], subject="multi-span coverage reading"
    )


def format_report(report: RoutingCalibrationReport) -> str:
    """Render tuning grid first and the one frozen final result second."""
    lines = [
        "# Fusion routing heuristic calibration",
        "",
        f"- tuning split: `{report['tuning_split']}`",
        f"- held-out final split: `{report['final_split']}`",
        "- sidecar visible to router: `no`",
        f"- graph row: `{report['graph_strategy']}@{report['graph_weight']:.2f}/"
        f"d{report['candidates']}/i{report['span_identity']}`",
        f"- bootstrap: {report['resamples']} resamples, seed {report['seed']}",
        f"- frozen policy: `{report['frozen_policy']}`",
        f"- recommended policy: `{report['recommended_policy'] or 'none'}`",
        f"- decision: **{report['decision']}** -- {report['reason']}",
        "",
        "Tuning selects one policy by requiring its multi-span coverage interval to clear zero",
        "without a single-span recall interval below zero. Final is evaluated only for that",
        "frozen policy and must independently pass the same gate before recommendation.",
        "",
        "`d` is the number of multi-span questions the routed and vector lanes differ on. The",
        "coverage half of the gate is a separation claim, so it also requires enough differing",
        "questions for the reporting level to be reachable; the single-span half is a",
        "non-inferiority check and is read on the interval alone.",
        "",
        "## Tuning threshold grid",
        "",
    ]
    lines += _gate_summary(report)
    lines += _table(report["tuning"])
    lines += ["## Frozen policy on final", ""]
    lines += _table({report["frozen_policy"]: report["final"]})
    lines += ["## Frozen final routing errors", ""]
    errors = report["final"]["route_errors"]
    if not errors:
        lines += ["None.", ""]
    else:
        lines += ["| item | predicted | expected | signals |", "| --- | --- | --- | --- |"]
        lines += [
            f"| `{error['item_id']}` | {error['predicted']} | {error['expected']} "
            f"| {', '.join(error['signals']) or 'none'} |"
            for error in errors
        ]
        lines.append("")
    return "\n".join(lines)
