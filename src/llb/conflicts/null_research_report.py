"""Durable JSON and Markdown artifacts for conflict-null research."""

import json
from pathlib import Path
from typing import Any

from llb.core.contracts.common import JsonObject

NULL_RESEARCH_SUMMARY = "summary.json"
NULL_RESEARCH_REPORT = "report.md"

_METHOD_LIMITATIONS = {
    "cross_corpus": (
        "The reference pairs are independently unrelated, but corpus/domain shift can make this "
        "null easier than the target population."
    ),
    "token_permutation": (
        "Token shuffling destroys order but preserves vocabulary; retrieval encoders can remain "
        "close to the original chunk."
    ),
    "sentence_permutation": (
        "Sentence shuffling preserves every local sentence and often most of the encoded meaning."
    ),
    "held_out_document": (
        "The document-pair maxima come from the observed corpus, contain real relations, and are "
        "not independent."
    ),
    "labelled_calibration": (
        "This is a supervised operating point on the planted fixture, not a null distribution or "
        "a transferable FPR estimate."
    ),
}


def _method_rows(methods: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| method | fixture P/R/F1 | HR recovered | goods rows | tail resolved | accepted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method in methods:
        fixture = method["fixture"]
        hr = method["hr"]
        goods = method["goods"]
        gates = method["gates"]
        rows.append(
            "| {method} | {precision:.3f}/{recall:.3f}/{f1:.3f} | {recovered}/{baseline} | "
            "{goods_rows} | {tail} | {accepted} |".format(
                method=method["method"],
                precision=fixture["precision"],
                recall=fixture["recall"],
                f1=fixture["f1"],
                recovered=hr["baseline_recovered"],
                baseline=hr["baseline_chunk_pairs"],
                goods_rows=goods["selected_chunk_pairs"],
                tail="yes" if gates["tail_resolved"] else "no",
                accepted="yes" if gates["accepted"] else "no",
            )
        )
    return rows


def render_null_research(summary: JsonObject) -> str:
    methods = summary["methods"]
    assert isinstance(methods, list)
    typed_methods = [method for method in methods if isinstance(method, dict)]
    rank = summary["rank_baseline"]
    parameters = summary["parameters"]
    datasets = summary["datasets"]
    assert isinstance(rank, dict)
    assert isinstance(parameters, dict)
    assert isinstance(datasets, dict)
    lines = [
        "# Corpus conflict independent-null research",
        "",
        f"- verdict: **{summary['verdict']}**",
        f"- nominal null FPR: {parameters['fpr']}",
        f"- fixture rank baseline: budget {rank['budget']}, "
        f"P/R/F1 {rank['precision']:.3f}/{rank['recall']:.3f}/{rank['f1']:.3f}",
        "",
        "## Dataset geometry",
        "",
        "| dataset | docs | chunks | comparable chunks | comparable pairs | centered |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in datasets.items():
        if not isinstance(payload, dict):
            continue
        lines.append(
            f"| {name} | {payload['documents']} | {payload['chunks']} | "
            f"{payload['comparable_chunks']} | {payload['comparable_chunk_pairs']} | "
            f"{'yes' if payload['centered'] else 'no'} |"
        )
    lines.extend(["", "## Acceptance matrix", "", *_method_rows(typed_methods), ""])
    for method in typed_methods:
        gates = method["gates"]
        failed = [name for name, passed in gates.items() if name != "accepted" and not passed]
        lines.extend(
            [
                f"### {method['method']}",
                "",
                f"- resolved thresholds: {json.dumps(method['thresholds'], sort_keys=True)}",
                f"- failed gates: {', '.join(failed) if failed else 'none'}",
                f"- limitation: {_METHOD_LIMITATIONS[str(method['method'])]}",
                "",
            ]
        )
    if summary["verdict"] == "negative":
        lines.extend(
            [
                "## Decision",
                "",
                "No candidate satisfies all gates. The semantic tier remains a ranked candidate "
                "generator and must not quote a false-positive rate. Threshold selection should "
                "move toward claim-tier measured precision before another semantic default is "
                "considered.",
                "",
            ]
        )
    return "\n".join(lines)


def write_null_research(out_dir: Path, summary: JsonObject) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / NULL_RESEARCH_SUMMARY
    report_path = out_dir / NULL_RESEARCH_REPORT
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_null_research(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}
