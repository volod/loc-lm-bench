"""Paired raw-to-prepared conversion counts for the per-hop probe."""

from collections.abc import Sequence

from llb.rag.multihop_probe.models import (
    DIAGNOSES,
    DiagnosisCohortConversion,
    ItemBudgetOutcome,
    ItemProbe,
    MultiHopProbeReport,
    QueryPrepConversion,
)


def _operating_outcome(probe: ItemProbe) -> ItemBudgetOutcome:
    return probe["budgets"][0]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _change_counts(pairs: Sequence[tuple[float, float]]) -> tuple[int, int, int]:
    """Count improvement, tie, and regression for one paired numeric measure."""
    return (
        sum(after > before for before, after in pairs),
        sum(after == before for before, after in pairs),
        sum(after < before for before, after in pairs),
    )


def _all_spans_counts(pairs: Sequence[tuple[float, float]]) -> tuple[int, int, int, int]:
    """Count raw/prepared successes and paired binary gains/losses."""
    return (
        sum(before == 1.0 for before, _after in pairs),
        sum(after == 1.0 for _before, after in pairs),
        sum(before == 0.0 and after == 1.0 for before, after in pairs),
        sum(before == 1.0 and after == 0.0 for before, after in pairs),
    )


def _reachability_counts(pairs: Sequence[tuple[ItemProbe, ItemProbe]]) -> tuple[int, int]:
    """Count items that gain or lose complete deep-pass reachability."""
    return (
        sum(
            before["limiting_rank"] is None and after["limiting_rank"] is not None
            for before, after in pairs
        ),
        sum(
            before["limiting_rank"] is not None and after["limiting_rank"] is None
            for before, after in pairs
        ),
    )


def _cohort(
    pairs: Sequence[tuple[ItemProbe, ItemProbe]], diagnosis: str
) -> DiagnosisCohortConversion:
    selected = [(before, after) for before, after in pairs if before["diagnosis"] == diagnosis]
    before_outcomes = [_operating_outcome(before) for before, _after in selected]
    after_outcomes = [_operating_outcome(after) for _before, after in selected]
    coverage_pairs = [
        (float(before["span_coverage"]), float(after["span_coverage"]))
        for before, after in zip(before_outcomes, after_outcomes, strict=True)
    ]
    all_spans_pairs = [
        (float(before["all_spans_at_k"]), float(after["all_spans_at_k"]))
        for before, after in zip(before_outcomes, after_outcomes, strict=True)
    ]
    all_before, all_after, all_gained, all_lost = _all_spans_counts(all_spans_pairs)
    coverage_improved, coverage_tied, coverage_regressed = _change_counts(coverage_pairs)
    newly_reachable, no_longer_reachable = _reachability_counts(selected)
    return {
        "n": len(selected),
        "all_spans_before": all_before,
        "all_spans_after": all_after,
        "all_spans_gained": all_gained,
        "all_spans_lost": all_lost,
        "span_coverage_before": _mean([before for before, _after in coverage_pairs]),
        "span_coverage_after": _mean([after for _before, after in coverage_pairs]),
        "span_coverage_improved": coverage_improved,
        "span_coverage_tied": coverage_tied,
        "span_coverage_regressed": coverage_regressed,
        "newly_reachable_at_depth": newly_reachable,
        "no_longer_reachable_at_depth": no_longer_reachable,
    }


def _paired_items(
    baseline: MultiHopProbeReport, prepared: MultiHopProbeReport
) -> list[tuple[ItemProbe, ItemProbe]]:
    prepared_by_id = {probe["item_id"]: probe for probe in prepared["items"]}
    if set(prepared_by_id) != {probe["item_id"] for probe in baseline["items"]}:
        raise ValueError("baseline and prepared focus-slice item ids differ")
    return [(probe, prepared_by_id[probe["item_id"]]) for probe in baseline["items"]]


def query_prep_conversion(
    baseline: MultiHopProbeReport, prepared: MultiHopProbeReport
) -> QueryPrepConversion:
    """Summarize conversion and cost by the RAW-query diagnosis cohort."""
    if baseline["focus_slice"] != prepared["focus_slice"]:
        raise ValueError("baseline and prepared focus slices differ")
    if baseline["budgets"] != prepared["budgets"]:
        raise ValueError("baseline and prepared retrieval budgets differ")
    pairs = _paired_items(baseline, prepared)
    transitions = {before: dict.fromkeys(DIAGNOSES, 0) for before in DIAGNOSES}
    for before, after in pairs:
        transitions[before["diagnosis"]][after["diagnosis"]] += 1
    return {
        "focus_slice": baseline["focus_slice"],
        "n": len(pairs),
        "operating_budget": baseline["budgets"][0],
        "cohorts": {diagnosis: _cohort(pairs, diagnosis) for diagnosis in DIAGNOSES},
        "transitions": transitions,
    }
