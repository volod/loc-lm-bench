"""The three numbers the context ablation exists to produce (pure).

Per-lane means answer "how well did the model do under this lane". They do not answer the
operator's question, which is a DIFFERENCE: how much of the RAG score did retrieval pay for
(`rag - closed_book`), and does whole-document stuffing beat chunked retrieval within the model's
window (`long_context - rag`). Both are paired per item, so the small-sample interval keeps the
per-item pairing that makes a few dozen items readable at all.

The contamination flag is the honesty check on the first number: an item the model answers with no
context at all was never a retrieval problem, and a corpus full of them makes any retrieval uplift
look small for reasons that have nothing to do with retrieval.

Every delta here is stated over a POPULATION of item positions rather than over "the run": the same
builder produces the pooled table and each question-type slice's own table, so a slice cannot drift
into reporting the same delta a slightly different way than the pool it came from.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.eval import common as eval_common
from llb.eval.context_ablation.models import (
    CONTAMINATION_COLUMNS,
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_ORACLE_DOCUMENT_GAP,
    DERIVED_ORACLE_DOCUMENT_GAP_FITTING,
    DERIVED_RETRIEVAL_UPLIFT,
    DERIVED_RETRIEVED_DOCUMENT_DELTA,
    DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING,
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    LANE_RETRIEVED_DOCUMENT,
    METRIC_OBJECTIVE,
    ContaminationReport,
    DerivedComparison,
)
from llb.eval.paired_cases import CaseRows, rows_by_item
from llb.rag.fusion_evidence.slices import MetricVectors
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, bootstrap_index_sets
from llb.rag.fusion_evidence.paired import paired_comparison

POPULATION_ALL = "all"
POPULATION_FITTING = "fitting"


def skipped_item_ids(rows: CaseRows) -> list[str]:
    """Items this lane skipped instead of scoring -- the context did not fit, and was not cut."""
    return sorted(
        str(row["item_id"])
        for row in rows
        if str(row.get("status", "")) == eval_common.CONTEXT_OVERFLOW
    )


def fitting_indexes(
    item_ids: Sequence[str], skipped: set[str], indexes: Sequence[int] | None = None
) -> list[int]:
    """Positions within `indexes` (the whole run by default) that the delta's own lanes scored."""
    population = range(len(item_ids)) if indexes is None else indexes
    return [i for i in population if item_ids[i] not in skipped]


def pair_skipped(skipped_by_lane: Mapping[str, Sequence[str]], *lanes: str) -> set[str]:
    """Items either lane of ONE paired delta skipped.

    Scoped to the pair, not to the run: with two document lanes present, an item only
    `retrieved_document` skipped says nothing about the `long_context - rag` population, and
    pooling every lane's skips would silently shrink a delta that was fully measured.
    """
    return {item_id for lane in lanes for item_id in skipped_by_lane.get(lane, ())}


def derived_comparison(
    label: str,
    *,
    candidate: str,
    reference: str,
    by_lane: Mapping[str, MetricVectors],
    indexes: Sequence[int],
    index_sets: list[list[int]],
    confidence: float = DEFAULT_CONFIDENCE,
    population: str = POPULATION_ALL,
    metric: str = METRIC_OBJECTIVE,
) -> DerivedComparison:
    """One paired candidate-minus-reference delta restricted to `indexes`."""
    candidate_values = [by_lane[candidate][metric][i] for i in indexes]
    reference_values = [by_lane[reference][metric][i] for i in indexes]
    return {
        "label": label,
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "n": len(indexes),
        "population": population,
        "paired": paired_comparison(candidate_values, reference_values, index_sets, confidence),
    }


def is_contaminated(row: Mapping[str, Any]) -> bool:
    """True when the closed-book answer already matches the reference.

    "Matches" is the canonical `run-eval` answer-side signal: the normalized strings are identical
    (`exact`), or every reference token appears in the answer (`contains`). Both are strict enough
    that a fluent near-miss does not qualify.
    """
    return any(float(row.get(column, 0.0) or 0.0) >= 1.0 for column in CONTAMINATION_COLUMNS)


def contamination_report(lane: str, rows: CaseRows, item_ids: Sequence[str]) -> ContaminationReport:
    """Which items the closed-book lane already answers, and how many that is."""
    by_item = rows_by_item(rows)
    flagged = [item_id for item_id in item_ids if is_contaminated(by_item.get(item_id, {}))]
    total = len(item_ids)
    return {
        "lane": lane,
        "n": total,
        "n_contaminated": len(flagged),
        "rate": round(len(flagged) / total, 4) if total else 0.0,
        "item_ids": flagged,
    }


# The paired deltas the report states, in reading order: what retrieval bought, what the oracle
# document lane adds over chunks, how much of that a RETRIEVED document captures, and what is left
# that only the gold label could have supplied. Each entry is
# `(label, fitting label or None, candidate lane, reference lane)`; a delta whose lanes were not
# both scored is simply absent, so a two- or three-lane selection reports what it can measure.
DELTAS = (
    (DERIVED_RETRIEVAL_UPLIFT, None, LANE_RAG, LANE_CLOSED_BOOK),
    (DERIVED_LONG_CONTEXT_DELTA, DERIVED_LONG_CONTEXT_DELTA_FITTING, LANE_LONG_CONTEXT, LANE_RAG),
    (
        DERIVED_RETRIEVED_DOCUMENT_DELTA,
        DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING,
        LANE_RETRIEVED_DOCUMENT,
        LANE_RAG,
    ),
    (
        DERIVED_ORACLE_DOCUMENT_GAP,
        DERIVED_ORACLE_DOCUMENT_GAP_FITTING,
        LANE_LONG_CONTEXT,
        LANE_RETRIEVED_DOCUMENT,
    ),
)


def _delta_entries(
    label: str,
    fitting_label: str | None,
    candidate: str,
    reference: str,
    *,
    by_lane: Mapping[str, MetricVectors],
    item_ids: Sequence[str],
    skipped_by_lane: Mapping[str, list[str]],
    indexes: Sequence[int],
    index_sets: list[list[int]],
    confidence: float,
    resamples: int,
    seed: int,
) -> list[DerivedComparison]:
    """One delta over the population, plus its fitting cut when either of its own lanes skipped."""
    if candidate not in by_lane or reference not in by_lane:
        return []
    entries = [
        derived_comparison(
            label,
            candidate=candidate,
            reference=reference,
            by_lane=by_lane,
            indexes=indexes,
            index_sets=index_sets,
            confidence=confidence,
        )
    ]
    skipped = pair_skipped(skipped_by_lane, candidate, reference)
    if fitting_label is None or not skipped:
        return entries
    fitting = fitting_indexes(item_ids, skipped, indexes)
    if len(fitting) == len(indexes):
        return entries
    entries.append(
        derived_comparison(
            fitting_label,
            candidate=candidate,
            reference=reference,
            by_lane=by_lane,
            indexes=fitting,
            index_sets=bootstrap_index_sets(len(fitting), resamples, seed),
            confidence=confidence,
            population=POPULATION_FITTING,
        )
    )
    return entries


def paired_deltas(
    by_lane: Mapping[str, MetricVectors],
    item_ids: Sequence[str],
    skipped_by_lane: Mapping[str, list[str]],
    indexes: Sequence[int],
    index_sets: list[list[int]],
    confidence: float,
    resamples: int,
    seed: int,
) -> list[DerivedComparison]:
    """Every measurable paired delta over one population of item positions."""
    entries: list[DerivedComparison] = []
    for label, fitting_label, candidate, reference in DELTAS:
        entries.extend(
            _delta_entries(
                label,
                fitting_label,
                candidate,
                reference,
                by_lane=by_lane,
                item_ids=item_ids,
                skipped_by_lane=skipped_by_lane,
                indexes=indexes,
                index_sets=index_sets,
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            )
        )
    return entries
