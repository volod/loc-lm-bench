"""Durable JSON and Markdown artifacts for conflict-null research."""

import json
from pathlib import Path
from typing import Any

from llb.conflicts.null_research_report_third import (
    decision_evidence,
    feasibility_section,
    identifiability_section,
    precision_section,
    role_section,
)
from llb.core.contracts.common import JsonObject

NULL_RESEARCH_SUMMARY = "summary.json"
NULL_RESEARCH_REPORT = "report.md"
NULL_RESEARCH_CONTROL_TRACES = "counterfactual_traces.jsonl"

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
    "surface_matched_reference": (
        "Matching reduces measured surface shift but reuses source and reference chunks, so pair "
        "rows are not independent tail observations."
    ),
    "surface_matched_residual": (
        "Subtracting each chunk's matched-control median is a local geometry correction; it is "
        "valid only when the matched controls pass exchangeability and clustered-tail gates."
    ),
    "surface_matched_cluster_fdr": (
        "The FDR numerator uses a conservative source-cluster Wilson bound; a threshold is "
        "unidentified when the effective control bank cannot bound the requested tail."
    ),
    "claim_counterfactual_control": (
        "Traceable capitalized-argument, quantity, and modality edits preserve topic and form, "
        "but may create a real conflict relation and lack independent semantic verification, so "
        "they are not an independent null."
    ),
    "balanced_transport_control": (
        "Cross-fitted propensity weighting uses every reference claim once, but a balanced "
        "covariate distribution does not make the cosine scale exchangeable, and the bank still "
        "bounds how fine a tail can be certified."
    ),
    "whitened_cosine": (
        "Whitening equalizes encoder axis variance; it rescales every pair, so its absolute "
        "cosines are not comparable to the shipped threshold and recovery must be read per pair."
    ),
    "anisotropy_stripped_cosine": (
        "Removing several leading directions removes more corpus shift than mean centering, and "
        "also removes whatever topical signal those directions carried."
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


def _dataset_rows(datasets: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for name, payload in datasets.items():
        if isinstance(payload, dict):
            rows.append(
                f"| {name} | {payload['documents']} | {payload['chunks']} | "
                f"{payload['comparable_chunks']} | {payload['comparable_chunk_pairs']} | "
                f"{'yes' if payload['centered'] else 'no'} |"
            )
    return rows


def _diagnostic_rows(method: dict[str, Any]) -> list[str]:
    diagnostics = method.get("diagnostics")
    tails = method.get("null_tails")
    if not isinstance(diagnostics, dict) or not isinstance(tails, dict):
        return []
    rows = [
        "| dataset | effective units | membership AUC | control valid |",
        "| --- | --- | --- | --- |",
    ]
    for dataset, payload in diagnostics.items():
        if not isinstance(payload, dict):
            continue
        tail = tails.get(dataset, {})
        effective = tail.get("effective_independent_units", "n/a")
        auc = payload.get("membership_auc", "n/a")
        valid = payload.get("exchangeable", payload.get("independent_semantic_verification", False))
        rows.append(f"| {dataset} | {effective} | {auc} | {'yes' if valid else 'no'} |")
    return [*rows, ""]


def _fdr_lines(method: dict[str, Any]) -> list[str]:
    fdr_fits = method.get("fdr_fits")
    if not isinstance(fdr_fits, dict):
        return []
    identified = ", ".join(
        f"{dataset}={'yes' if payload.get('identified') else 'no'}"
        for dataset, payload in fdr_fits.items()
        if isinstance(payload, dict)
    )
    return [f"- nonempty FDR point identified: {identified}", ""]


def _method_section(method: dict[str, Any]) -> list[str]:
    gates = method["gates"]
    failed = [name for name, passed in gates.items() if name != "accepted" and not passed]
    return [
        f"### {method['method']}",
        "",
        f"- resolved thresholds: {json.dumps(method['thresholds'], sort_keys=True)}",
        f"- failed gates: {', '.join(failed) if failed else 'none'}",
        f"- limitation: {_METHOD_LIMITATIONS[str(method['method'])]}",
        "",
        *_diagnostic_rows(method),
        *_fdr_lines(method),
    ]


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
    lines.extend(_dataset_rows(datasets))
    lines.extend(["", *feasibility_section(summary)])
    lines.extend(["## Acceptance matrix", "", *_method_rows(typed_methods), ""])
    for method in typed_methods:
        lines.extend(_method_section(method))
    lines.extend(identifiability_section(summary))
    lines.extend(precision_section(summary))
    lines.extend(role_section(summary))
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
                *decision_evidence(summary),
            ]
        )
    return "\n".join(lines)


def write_null_research(out_dir: Path, summary: JsonObject) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / NULL_RESEARCH_SUMMARY
    report_path = out_dir / NULL_RESEARCH_REPORT
    persisted = dict(summary)
    traces = persisted.pop("control_traces", None)
    paths = {"summary": summary_path, "report": report_path}
    if isinstance(traces, list):
        trace_path = out_dir / NULL_RESEARCH_CONTROL_TRACES
        trace_path.write_text(
            "".join(
                json.dumps(trace, sort_keys=True, ensure_ascii=True) + "\n" for trace in traces
            ),
            encoding="utf-8",
        )
        persisted["control_trace_artifact"] = NULL_RESEARCH_CONTROL_TRACES
        persisted["control_trace_rows"] = len(traces)
        paths["control_traces"] = trace_path
    summary_path.write_text(
        json.dumps(persisted, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_null_research(summary), encoding="utf-8")
    return paths
