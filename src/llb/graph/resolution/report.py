"""Render the resolution summary as the Markdown artifact an operator reads first.

Pure formatting over the summary payload, so the numbers a reader sees and the numbers
`summary.json` carries cannot drift apart.
"""

from llb.core.contracts.common import JsonObject
from llb.graph.resolution.verdict import DECISION_RECOMMEND, LANE_METRICS

_HEADER = "# Graph entity node resolution"


def format_resolution_report(summary: JsonObject) -> str:
    """The whole artifact: what was fitted, what each cut merged, and what it decided."""
    if summary.get("declined"):
        return "\n".join(
            [_HEADER, "", f"Not run: {summary['reason']}", f"Nodes: {summary['n_nodes']}", ""]
        )
    lines = [_HEADER, "", *_context(summary), "", *_baseline(summary), ""]
    lines.extend(_thresholds(summary))
    lines.extend(["", *_verdict(summary), ""])
    return "\n".join(lines)


def _context(summary: JsonObject) -> list[str]:
    linkage = summary.get("linkage", {})
    untrained = linkage.get("untrained_levels", [])
    lines = [
        f"Graph: {summary['n_nodes']} nodes, {summary['n_edges']} edges. "
        f"Items: {summary['n_items']} at k={summary['k']}.",
        "",
        f"Linkage: {linkage.get('n_scored_pairs', 0)} scored pairs, "
        f"{linkage.get('n_matched_pairs', 0)} at or above the loosest priced cut "
        f"({linkage.get('match_threshold')}), seed {linkage.get('seed')}.",
    ]
    if untrained:
        lines.extend(
            [
                "",
                f"Untrained comparison level(s): {', '.join(untrained)}. The exact-match level "
                "on `name` is unreachable by construction -- the graph builder already keys a "
                "node on its normalized name, so two records with an equal name would be one "
                "node.",
            ]
        )
    return lines


def _baseline(summary: JsonObject) -> list[str]:
    lines = ["## Pre-overlay lanes", "", "| Strategy | recall@k | MRR |", "| --- | --- | --- |"]
    lines.extend(
        f"| {strategy} | {_metric(row, 'recall_at_k')} | {_metric(row, 'mrr')} |"
        for strategy, row in sorted(summary.get("baseline", {}).items())
    )
    return lines


def _thresholds(summary: JsonObject) -> list[str]:
    lines = [
        "## Candidate cuts",
        "",
        "| Threshold | Clusters | Nodes merged | Largest | Strategy | recall@k | d recall | MRR | d MRR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.get("thresholds", []):
        for strategy, scored in sorted(row.get("strategies", {}).items()):
            lines.append(
                f"| {row['threshold']:g} | {row['n_clusters']} | {row['n_nodes_merged']} | "
                f"{row['largest_cluster']} | {strategy} | "
                f"{_metric(scored, 'recall_at_k')} | {_delta(scored, 'recall_at_k')} | "
                f"{_metric(scored, 'mrr')} | {_delta(scored, 'mrr')} |"
            )
    return lines


def _verdict(summary: JsonObject) -> list[str]:
    verdict = summary.get("verdict", {})
    decision = verdict.get("decision", "unknown")
    headline = (
        f"Recommended overlay lane: `{verdict.get('lane')}`"
        if decision == DECISION_RECOMMEND
        else "Negative result: no overlay is adopted"
    )
    return [
        "## Verdict",
        "",
        f"**{decision}** -- {headline}",
        "",
        verdict.get("reason", ""),
        "",
        verdict.get("note", ""),
    ]


def _metric(row: JsonObject, metric: str) -> str:
    value = row.get(metric)
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "-"


def _delta(row: JsonObject, metric: str) -> str:
    value = row.get(f"delta_{metric}")
    return f"{float(value):+.3f}" if isinstance(value, (int, float)) else "-"


def format_console_summary(summary: JsonObject) -> str:
    """The short operator-readable report the CLI prints."""
    if summary.get("declined"):
        return f"[graph-resolution] not run: {summary['reason']}"
    lines = [
        f"[graph-resolution] {summary['n_nodes']} nodes, {summary['n_items']} items, "
        f"k={summary['k']}",
    ]
    for row in summary.get("thresholds", []):
        lines.append(
            f"[graph-resolution] cut {row['threshold']:g}: {row['n_clusters']} cluster(s), "
            f"{row['n_nodes_merged']} node(s) merged, largest {row['largest_cluster']}"
        )
        lines.extend(
            f"  - {strategy}: "
            + ", ".join(
                f"{metric} {_metric(scored, metric)} ({_delta(scored, metric)})"
                for metric in LANE_METRICS
            )
            for strategy, scored in sorted(row.get("strategies", {}).items())
        )
    verdict = summary.get("verdict", {})
    lines.append(f"[graph-resolution] {verdict.get('decision')}: {verdict.get('reason')}")
    lines.append(f"[graph-resolution] {verdict.get('note')}")
    return "\n".join(lines)
