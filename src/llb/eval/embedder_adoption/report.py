"""Render the adoption-bar sweep as ASCII Markdown (AGENTS.md: no Unicode, no box-drawing).

Two tables, in the order the question is asked. The DELTA table first -- one row per cell, the
candidate minus the baseline -- because that is what the bar decision reads. Then the per-lane
means, which is where an operator checks that a flat delta is two lanes agreeing rather than two
lanes both collapsing at a small `top_k`.
"""

from collections.abc import Mapping

from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    METRIC_TOKEN_F1,
)
from llb.eval.embedder_adoption.models import (
    METRIC_CONTAINS,
    METRIC_RECIPROCAL_RANK,
    AdoptionBarReport,
    CellReport,
)
from llb.rag.fusion_evidence.stats import format_interval

_HEADERS = {
    METRIC_OBJECTIVE: "objective",
    METRIC_CONTAINS: "found rate",
    METRIC_TOKEN_F1: "token F1",
    METRIC_RETRIEVAL_HIT: "recall@k",
    METRIC_RECIPROCAL_RANK: "MRR@k",
}
_RERANKER_OFF = "off"


def _cell_note(cell: CellReport) -> str:
    return f"top_k={cell['top_k']}, reranker={cell['reranker'] or _RERANKER_OFF}"


def _delta_table(report: AdoptionBarReport) -> list[str]:
    """One row per cell: the candidate-minus-baseline paired delta on every metric."""
    metrics = list(report["metrics"])
    lines = [
        "### Answer-side delta per cell",
        "",
        f"`{report['candidate']}` minus `{report['baseline']}` on the identical "
        f"{len(report['item_ids'])} items, {report['confidence']:.0%} paired bootstrap CI.",
        "",
        "| cell | config | "
        + " | ".join(f"d {_HEADERS.get(metric, metric)}" for metric in metrics)
        + " | objective w/l/t | sign p |",
        "| --- | --- | " + " | ".join(["---:"] * len(metrics)) + " | :-: | ---: |",
    ]
    for cell in report["cells"]:
        objective = cell["paired"][METRIC_OBJECTIVE]
        deltas = " | ".join(format_interval(cell["paired"][metric]["delta"]) for metric in metrics)
        lines.append(
            f"| `{cell['label']}` | {_cell_note(cell)} | {deltas} "
            f"| {objective['wins']}/{objective['losses']}/{objective['ties']} "
            f"| {objective['sign_test_p']:.3f} |"
        )
    lines.append("")
    return lines


def _lane_table(report: AdoptionBarReport) -> list[str]:
    """Both encoders' own means per cell -- the level the deltas above were taken at."""
    metrics = list(report["metrics"])
    lines = [
        "### Per-encoder means",
        "",
        "| cell | encoder | "
        + " | ".join(_HEADERS.get(metric, metric) for metric in metrics)
        + " |",
        "| --- | --- | " + " | ".join(["---:"] * len(metrics)) + " |",
    ]
    for cell in report["cells"]:
        for model in sorted(cell["lanes"]):
            entry = cell["lanes"][model]
            values = " | ".join(format_interval(entry["metrics"][metric]) for metric in metrics)
            lines.append(f"| `{cell['label']}` | `{model}` | {values} |")
    lines.append("")
    return lines


def _bundle_list(report: AdoptionBarReport) -> list[str]:
    """Every scored run bundle, so any cell is reproducible by re-running `run-eval` on it."""
    lines = ["### Scored run bundles", ""]
    for cell in report["cells"]:
        lines.append(f"- `{cell['label']}` ({_cell_note(cell)})")
        for model in sorted(cell["lanes"]):
            lines.append(f"  - `{model}`")
            lines.extend(f"    - `{run_dir}`" for run_dir in cell["lanes"][model]["run_dirs"])
    lines.append("")
    return lines


def format_report(
    report: AdoptionBarReport,
    *,
    metadata: Mapping[str, object] | None = None,
    title: str = "Embedder adoption bar: does a first-hit-rank gain reach the answer?",
) -> str:
    """The full Markdown artifact: the verdict, the per-cell deltas, the levels, the bundles."""
    verdict = report["verdict"]
    meta = dict(metadata or {})
    lines = [f"# {title}", ""]
    for key in ("model", "backend", "split", "grounding", "goldset", "corpus"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        f"- baseline encoder: `{report['baseline']}`",
        f"- candidate encoder: `{report['candidate']}`",
        f"- scored items: {len(report['item_ids'])} (identical set in every cell and lane)",
        "- cells: " + ", ".join(f"`{cell['label']}`" for cell in report["cells"]),
        f"- bootstrap: {report['resamples']} resamples, seed {report['seed']}",
        f"- verdict: **{verdict['decision']}** -- {verdict['reason']}",
        "",
    ]
    lines += _delta_table(report)
    lines += _lane_table(report)
    lines += _bundle_list(report)
    return "\n".join(lines) + "\n"


def format_summary(report: AdoptionBarReport) -> str:
    """One-screen ASCII summary for the terminal (the artifact carries the full tables)."""
    verdict = report["verdict"]
    lines = [
        f"[compare-embedder-adoption] n={len(report['item_ids'])} "
        f"candidate={report['candidate']} baseline={report['baseline']}",
        f"  {'cell'.ljust(12)} {'d objective':>22} {'d MRR@k':>22}",
    ]
    for cell in report["cells"]:
        lines.append(
            f"  {cell['label'].ljust(12)} "
            f"{format_interval(cell['paired'][METRIC_OBJECTIVE]['delta']):>22} "
            f"{format_interval(cell['paired'][METRIC_RECIPROCAL_RANK]['delta']):>22}"
        )
    lines.append(f"  verdict: {verdict['decision'].upper()} -- {verdict['reason']}")
    return "\n".join(lines)
