"""ASCII Markdown for the query-prep conversion extension of the per-hop probe."""

from llb.rag.multihop_probe.models import (
    DIAGNOSES,
    DiagnosisCohortConversion,
    ItemProbe,
    MultiHopProbeReport,
    MultiHopQueryPrepReport,
)

_UNREACHED = "-"


def _headline(report: MultiHopQueryPrepReport) -> list[str]:
    conversion = report["conversion"]
    query = conversion["cohorts"]["query"]
    budget = conversion["cohorts"]["budget"]
    k = conversion["operating_budget"]
    return [
        f"Query-diagnosed conversion: {query['all_spans_gained']}/{query['n']} items now carry "
        f"every hop at k={k}; {query['newly_reachable_at_depth']}/{query['n']} make every hop "
        "reachable in the deep pass.",
        "",
        f"Budget-diagnosed cost: {budget['span_coverage_regressed']}/{budget['n']} items lose "
        f"covered-span share at k={k}; {budget['span_coverage_improved']} improve and "
        f"{budget['span_coverage_tied']} tie.",
        "",
    ]


def _cohort_row(name: str, cohort: DiagnosisCohortConversion) -> str:
    return (
        f"| {name} | {cohort['n']} | {cohort['all_spans_before']} -> "
        f"{cohort['all_spans_after']} | +{cohort['all_spans_gained']} / "
        f"-{cohort['all_spans_lost']} | {cohort['span_coverage_before']:.3f} -> "
        f"{cohort['span_coverage_after']:.3f} | {cohort['span_coverage_improved']} / "
        f"{cohort['span_coverage_tied']} / {cohort['span_coverage_regressed']} | "
        f"+{cohort['newly_reachable_at_depth']} / "
        f"-{cohort['no_longer_reachable_at_depth']} |"
    )


def _cohort_table(report: MultiHopQueryPrepReport) -> list[str]:
    cohorts = report["conversion"]["cohorts"]
    return [
        "## Conversion by raw-query diagnosis",
        "",
        "`all-spans +/-` counts operating-budget item conversions/regressions. "
        "`coverage +/-/=` counts items whose covered-span SHARE improves/regresses/ties. "
        "`deep +/-` counts items that newly reach/lose every hop at probe depth.",
        "",
        "| baseline diagnosis | n | all-spans before -> after | all-spans +/- | "
        "span coverage before -> after | coverage +/=/- | deep +/- |",
        "| --- | ---: | :-: | :-: | :-: | :-: | :-: |",
        *(_cohort_row(name, cohorts[name]) for name in DIAGNOSES),
        "",
    ]


def _curve_table(baseline: MultiHopProbeReport, prepared: MultiHopProbeReport) -> list[str]:
    focus = baseline["focus_slice"]
    before = baseline["slices"][focus]["curve"]
    after = prepared["slices"][focus]["curve"]
    lines = [
        f"## Focus-slice curve ({focus})",
        "",
        "| k | raw all-spans@k | prepared all-spans@k | raw span coverage | "
        "prepared span coverage |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for raw_point, prepared_point in zip(before, after, strict=True):
        lines.append(
            f"| {raw_point['k']} | {raw_point['all_spans_at_k']['mean']:.3f} | "
            f"{prepared_point['all_spans_at_k']['mean']:.3f} | "
            f"{raw_point['span_coverage']:.3f} | {prepared_point['span_coverage']:.3f} |"
        )
    lines.append("")
    return lines


def _transition_table(report: MultiHopQueryPrepReport) -> list[str]:
    transitions = report["conversion"]["transitions"]
    rows = [
        f"| {before} | {after} | {count} |"
        for before in DIAGNOSES
        for after, count in transitions[before].items()
        if count
    ]
    return [
        "## Diagnosis transitions",
        "",
        "| raw diagnosis | prepared diagnosis | items |",
        "| --- | --- | ---: |",
        *rows,
        "",
    ]


def _rank_list(probe: ItemProbe) -> str:
    return " / ".join(
        _UNREACHED if hop["question_rank"] is None else str(hop["question_rank"])
        for hop in probe["hops"]
    )


def _item_ledger(report: MultiHopQueryPrepReport) -> list[str]:
    prepared = {probe["item_id"]: probe for probe in report["prepared"]["items"]}
    lines = [
        "## Paired item ledger",
        "",
        "Generated decomposition text and subqueries are retained in `probe.json`; this ASCII "
        "ledger records their count and the ranks they produced.",
        "",
        "| item | diagnosis raw -> prepared | all-spans raw/prepared | "
        "span coverage raw/prepared | deep ranks raw -> prepared | subqueries |",
        "| --- | --- | :-: | :-: | --- | ---: |",
    ]
    for before in report["baseline"]["items"]:
        after = prepared[before["item_id"]]
        before_at_k = before["budgets"][0]
        after_at_k = after["budgets"][0]
        provenance = after.get("query_prep", {})
        subqueries = provenance.get("query_subqueries", [])
        n_subqueries = len(subqueries) if isinstance(subqueries, list) else 0
        lines.append(
            f"| {before['item_id']} | {before['diagnosis']} -> {after['diagnosis']} | "
            f"{before_at_k['all_spans_at_k']:.0f}/{after_at_k['all_spans_at_k']:.0f} | "
            f"{before_at_k['span_coverage']:.3f}/{after_at_k['span_coverage']:.3f} | "
            f"{_rank_list(before)} -> {_rank_list(after)} | {n_subqueries} |"
        )
    lines.append("")
    return lines


def format_query_prep_probe_report(report: MultiHopQueryPrepReport) -> str:
    """Render the paired raw/prepared probe and its diagnosis-cohort reading."""
    baseline = report["baseline"]
    endpoint = report.get("endpoint")
    endpoint_line = (
        f", endpoint `{endpoint['backend']}:{endpoint['model']}`" if endpoint is not None else ""
    )
    lines = [
        "# Multi-hop query-prep conversion probe",
        "",
        f"Lane `{baseline['lane']}`, query prep `{','.join(report['query_prep_steps'])}`"
        f"{endpoint_line}, {report['conversion']['n']} focus-slice items, "
        f"probe depth {baseline['probe_depth']}.",
        "",
        *_headline(report),
        *_cohort_table(report),
        *_curve_table(baseline, report["prepared"]),
        *_transition_table(report),
        *_item_ledger(report),
    ]
    return "\n".join(lines).rstrip() + "\n"
