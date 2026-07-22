"""Render the answer-quality comparison as ASCII Markdown (AGENTS.md: no Unicode, no box-drawing).

Same three-table shape as the fusion-evidence report, so the two artifacts read side by side: the
focus slice first (did the answers get better?), then overall (did anything else pay for it?), then
the item-level ledger, which at a few dozen items IS the evidence.
"""

from collections.abc import Mapping

from llb.eval.answer_quality.models import (
    METRIC_ALL_SPANS,
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    METRIC_SPAN_COVERAGE,
    METRIC_TOKEN_F1,
    AnswerQualityReport,
)
from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion_evidence.models import ROUTED_ROW_PREFIX
from llb.rag.fusion_evidence.stats import format_interval

_HEADERS = {
    METRIC_OBJECTIVE: "objective",
    METRIC_TOKEN_F1: "token F1",
    METRIC_RETRIEVAL_HIT: "recall@k",
    METRIC_ALL_SPANS: "all-spans@k",
    METRIC_SPAN_COVERAGE: "span coverage",
}
_FACTOID_SLICE = "factoid"


def _headline_metrics(report: AnswerQualityReport) -> list[str]:
    return list(report["metrics"])


def _metric_table(
    report: AnswerQualityReport, pick: str | None, title: str, note: str
) -> list[str]:
    """One row-per-lane table of `mean [lo, hi]` per metric, plus the paired objective delta."""
    lines = [f"### {title}", ""]
    selected: dict[str, SliceReport] = {}
    for label, lane in report["lanes"].items():
        slice_report = lane["overall"] if pick is None else lane["slices"].get(pick)
        if slice_report is not None:
            selected[label] = slice_report
    n = next((entry["n"] for entry in selected.values()), 0)
    lines.append(f"{note} (n={n}, {report['confidence']:.0%} bootstrap CI)")
    lines.append("")
    if n == 0:
        lines.extend(["No item falls in this slice, so no metric is measured here.", ""])
        return lines
    metrics = _headline_metrics(report)
    header = " | ".join(_HEADERS.get(metric, metric) for metric in metrics)
    lines.append(f"| lane | {header} | objective delta vs {report['baseline']} | w/l/t | sign p |")
    lines.append("| --- | " + " | ".join(["---:"] * len(metrics)) + " | ---: | :-: | ---: |")
    for label in sorted(selected):
        entry = selected[label]
        cells = [format_interval(entry["metrics"][metric]) for metric in metrics]
        paired = entry["paired_vs_baseline"][METRIC_OBJECTIVE]
        lines.append(
            f"| {label} | "
            + " | ".join(cells)
            + f" | {format_interval(paired['delta'])} "
            + f"| {paired['wins']}/{paired['losses']}/{paired['ties']} "
            + f"| {paired['sign_test_p']:.3f} |"
        )
    lines.append("")
    return lines


def _item_table(report: AnswerQualityReport) -> list[str]:
    """Per-item objective / retrieval-hit outcome of every lane on the focus slice."""
    items = report["focus_items"]
    lines = [f"### Item-level outcomes ({report['focus_slice']})", ""]
    if not items:
        lines.extend([f"No {report['focus_slice']} item was scored.", ""])
        return lines
    coverage = report["verdict"]["coverage_metric"]
    labels = sorted(report["lanes"])
    lines.append("| item | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join([":-:"] * len(labels)) + " |")
    for item in items:
        cells = [
            f"{item['lanes'][label][METRIC_OBJECTIVE]:.2f}/{item['lanes'][label][coverage]:.2f}"
            for label in labels
        ]
        lines.append(f"| {item['item_id']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Each cell is `objective / {_HEADERS.get(coverage, coverage)}`. An item whose")
    lines.append("coverage rises while its objective does not is exactly the retrieval-only effect")
    lines.append("this lane exists to name.")
    lines.append("")
    return lines


def _lane_decisions(report: AnswerQualityReport) -> list[str]:
    """A decision per candidate lane -- the headline verdict only names the winner."""
    decisions = report["verdict"]["lane_decisions"]
    if len(decisions) < 2:
        return []
    lines = ["### Per-lane decisions", ""]
    for label in sorted(decisions):
        entry = decisions[label]
        lines.append(f"- `{label}`: **{entry['decision']}** -- {entry['reason']}")
    lines.append("")
    return lines


def _routing_outcomes(report: AnswerQualityReport) -> list[str]:
    """Plain-language safety reading for routed lanes: focus gain plus factoid passthrough."""
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
        factoid = lane["slices"].get(_FACTOID_SLICE)
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


def _lane_list(report: AnswerQualityReport) -> list[str]:
    """One entry per lane, naming the run bundle(s) its per-case scores came from."""
    lines = []
    for label, lane in sorted(report["lanes"].items()):
        lines.append(f"  - `{label}`")
        lines.extend(f"    - `{run_dir}`" for run_dir in lane["run_dirs"])
    return lines


def format_report(
    report: AnswerQualityReport,
    *,
    metadata: Mapping[str, object] | None = None,
    title: str = "Multi-hop answer quality",
) -> str:
    """The full Markdown artifact: verdict, focus slice, overall, other slices, item ledger."""
    verdict = report["verdict"]
    meta = dict(metadata or {})
    lines = [f"# {title}", ""]
    for key in ("model", "backend", "split", "grounding", "goldset"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        f"- baseline lane: `{report['baseline']}`",
        f"- scored items: {report['n']} (identical item set in every lane)",
        f"- focus slice: `{verdict['focus_slice']}` (n={verdict['focus_n']})",
        f"- bootstrap: {report['resamples']} resamples, seed {report['seed']}",
        f"- verdict: **{verdict['decision']}**"
        + (f" -- {verdict['reason']}" if verdict["reason"] else ""),
        "- scored lanes:",
    ]
    lines += _lane_list(report)
    lines.append("")
    lines += _lane_decisions(report)
    lines += _routing_outcomes(report)
    lines += _metric_table(
        report,
        report["focus_slice"],
        f"Focus slice: {report['focus_slice']}",
        "Questions whose answer needs evidence from more than one span",
    )
    lines += _metric_table(report, None, "Overall", "Every scored item")
    other = [
        name
        for lane in report["lanes"].values()
        for name in lane["slices"]
        if name != report["focus_slice"]
    ]
    for name in sorted(set(other)):
        lines += _metric_table(report, name, f"Slice: {name}", "Context slice")
    lines += _item_table(report)
    return "\n".join(lines) + "\n"
