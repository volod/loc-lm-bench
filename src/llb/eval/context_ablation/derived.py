"""The three numbers the context ablation exists to produce (pure).

Per-lane means answer "how well did the model do under this lane". They do not answer the
operator's question, which is a DIFFERENCE: how much of the RAG score did retrieval pay for
(`rag - closed_book`), and does whole-document stuffing beat chunked retrieval within the model's
window (`long_context - rag`). Both are paired per item, so the small-sample interval keeps the
per-item pairing that makes a few dozen items readable at all.

The contamination flag is the honesty check on the first number: an item the model answers with no
context at all was never a retrieval problem, and a corpus full of them makes any retrieval uplift
look small for reasons that have nothing to do with retrieval.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.eval import common as eval_common
from llb.eval.context_ablation.models import (
    CONTAMINATION_COLUMNS,
    METRIC_OBJECTIVE,
    ContaminationReport,
    DerivedComparison,
)
from llb.eval.paired_cases import CaseRows, rows_by_item
from llb.rag.fusion_evidence.slices import MetricVectors
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, paired_comparison

POPULATION_ALL = "all"
POPULATION_FITTING = "fitting"


def skipped_item_ids(rows: CaseRows) -> list[str]:
    """Items this lane skipped instead of scoring -- the context did not fit, and was not cut."""
    return sorted(
        str(row["item_id"])
        for row in rows
        if str(row.get("status", "")) == eval_common.CONTEXT_OVERFLOW
    )


def fitting_indexes(item_ids: Sequence[str], skipped: set[str]) -> list[int]:
    """Positions of the items no lane skipped."""
    return [i for i, item_id in enumerate(item_ids) if item_id not in skipped]


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
