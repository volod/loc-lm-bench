"""Turn per-hop ranks into the one reading the lane exists to produce.

The rule is deliberately small, because the whole value of the probe is that a reader can check
it: the WORST hop decides the item, and a hop the question never reaches is read against the same
hop queried by its own text. A hop the span text finds at the operating budget but the question
does not is a query problem; one neither form finds is not a ranking or budget question at all.
"""

from collections.abc import Sequence

from llb.rag.multihop_probe.models import (
    BUDGET_BUCKET_BEYOND,
    DIAGNOSIS_BUDGET,
    DIAGNOSIS_COVERED,
    DIAGNOSIS_QUERY,
    DIAGNOSIS_UNREACHABLE,
    DIAGNOSES,
    EXPLANATION_BUDGET,
    EXPLANATION_MIXED,
    EXPLANATION_NONE,
    EXPLANATION_QUERY,
    EXPLANATION_UNREACHABLE,
    DiagnosisReport,
    HopOutcome,
    ItemProbe,
)

# Which explanation each failing diagnosis supports (a covered item explains nothing).
_EXPLANATION_OF = {
    DIAGNOSIS_BUDGET: EXPLANATION_BUDGET,
    DIAGNOSIS_QUERY: EXPLANATION_QUERY,
    DIAGNOSIS_UNREACHABLE: EXPLANATION_UNREACHABLE,
}
_REASON_OF = {
    DIAGNOSIS_BUDGET: "the question reaches every hop, below the cut",
    DIAGNOSIS_QUERY: "only the span's own text reaches the hop at k={budget}",
    DIAGNOSIS_UNREACHABLE: "no query form reaches the hop at k={budget}",
}


def diagnose_item(hops: Sequence[HopOutcome], operating_budget: int) -> str:
    """Classify one item by its worst hop: covered, budget-limited, query-limited, or absent."""
    if all(_within(hop["question_rank"], operating_budget) for hop in hops):
        return DIAGNOSIS_COVERED
    unreached = [hop for hop in hops if hop["question_rank"] is None]
    if not unreached:
        return DIAGNOSIS_BUDGET
    if all(_within(hop["span_query_rank"], operating_budget) for hop in unreached):
        return DIAGNOSIS_QUERY
    return DIAGNOSIS_UNREACHABLE


def _within(rank: int | None, budget: int) -> bool:
    return rank is not None and rank <= budget


def item_min_budget(
    limiting_rank: int | None, budgets: Sequence[int], probe_depth: int
) -> int | str:
    """The smallest compared cutoff that carries EVERY hop, or `beyond` when none does."""
    if limiting_rank is None:
        return BUDGET_BUCKET_BEYOND
    for budget in (*budgets, probe_depth):
        if limiting_rank <= budget:
            return budget
    return BUDGET_BUCKET_BEYOND


def _budget_histogram(
    probes: Sequence[ItemProbe], budgets: Sequence[int], probe_depth: int
) -> dict[str, int]:
    """How many items each cutoff would carry both hops at -- the k question, as a histogram."""
    buckets = [str(budget) for budget in (*budgets, probe_depth)]
    histogram = dict.fromkeys([*dict.fromkeys(buckets), BUDGET_BUCKET_BEYOND], 0)
    for probe in probes:
        histogram[str(probe["min_budget"])] += 1
    return histogram


def _explanation(counts: dict[str, int], operating_budget: int) -> tuple[str, str]:
    """The explanation the counted failures support, and the sentence that states it."""
    failing = {name: counts[name] for name in _EXPLANATION_OF}
    total = sum(failing.values())
    if total == 0:
        return (
            EXPLANATION_NONE,
            f"every item carries all of its labeled spans at k={operating_budget}",
        )
    top = max(failing.values())
    leaders = [name for name, count in failing.items() if count == top]
    detail = ", ".join(
        f"{failing[name]} {name} ({_REASON_OF[name].format(budget=operating_budget)})"
        for name in _EXPLANATION_OF
    )
    head = f"of {total} {'item' if total == 1 else 'items'} missing a hop at k={operating_budget}"
    if len(leaders) > 1:
        return EXPLANATION_MIXED, f"{head}: {detail}; no single cause leads"
    return _EXPLANATION_OF[leaders[0]], f"{head}: {detail}"


def slice_diagnosis(
    probes: Sequence[ItemProbe], budgets: Sequence[int], probe_depth: int
) -> DiagnosisReport:
    """Count one slice's per-item diagnoses and name the explanation they support."""
    counts = {name: 0 for name in DIAGNOSES}
    for probe in probes:
        counts[probe["diagnosis"]] += 1
    explanation, reason = _explanation(counts, budgets[0])
    return {
        "n": len(probes),
        "counts": counts,
        "failing_items": len(probes) - counts[DIAGNOSIS_COVERED],
        "budget_histogram": _budget_histogram(probes, budgets, probe_depth),
        "explanation": explanation,
        "reason": reason,
    }
