"""Compare the three context lanes over one identical item set (pure).

File-driven like every other evidence lane: the input is one list of canonical per-case rows per
lane plus the question-type sidecar labels, so the whole comparison is unit-tested with dict rows
-- no backend, no store, no GPU. The per-lane slices reuse the shared bootstrap middle layer, so
this artifact reads beside the retrieval sweep and the answer-quality comparison; what is new is
the derived-delta table and the contamination flag.
"""

from collections.abc import Mapping, Sequence

from llb.eval.context_ablation.derived import (
    POPULATION_FITTING,
    contamination_report,
    derived_comparison,
    fitting_indexes,
    skipped_item_ids,
)
from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_RETRIEVAL_UPLIFT,
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    METRICS,
    ContextAblationReport,
    DerivedComparison,
    ItemOutcome,
    LaneReport,
)
from llb.eval.context_ablation.verdict import decide
from llb.eval.paired_cases import CaseRows, lane_vectors, shared_item_ids
from llb.rag.fusion_evidence.slices import (
    MetricVectors,
    slice_index_sets,
    slice_indexes,
    slice_report,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
)


def _derived(
    by_lane: Mapping[str, MetricVectors],
    item_ids: Sequence[str],
    skipped: set[str],
    index_sets: list[list[int]],
    confidence: float,
    resamples: int,
    seed: int,
) -> list[DerivedComparison]:
    """Retrieval uplift, the long-context delta, and -- when items were skipped -- its fitting cut."""
    all_indexes = list(range(len(item_ids)))
    entries: list[DerivedComparison] = []
    if LANE_RAG in by_lane and LANE_CLOSED_BOOK in by_lane:
        entries.append(
            derived_comparison(
                DERIVED_RETRIEVAL_UPLIFT,
                candidate=LANE_RAG,
                reference=LANE_CLOSED_BOOK,
                by_lane=by_lane,
                indexes=all_indexes,
                index_sets=index_sets,
                confidence=confidence,
            )
        )
    if LANE_LONG_CONTEXT not in by_lane or LANE_RAG not in by_lane:
        return entries
    entries.append(
        derived_comparison(
            DERIVED_LONG_CONTEXT_DELTA,
            candidate=LANE_LONG_CONTEXT,
            reference=LANE_RAG,
            by_lane=by_lane,
            indexes=all_indexes,
            index_sets=index_sets,
            confidence=confidence,
        )
    )
    if not skipped:
        return entries
    fitting = fitting_indexes(item_ids, skipped)
    entries.append(
        derived_comparison(
            DERIVED_LONG_CONTEXT_DELTA_FITTING,
            candidate=LANE_LONG_CONTEXT,
            reference=LANE_RAG,
            by_lane=by_lane,
            indexes=fitting,
            index_sets=bootstrap_index_sets(len(fitting), resamples, seed),
            confidence=confidence,
            population=POPULATION_FITTING,
        )
    )
    return entries


def _items(
    item_ids: Sequence[str],
    question_types: Mapping[str, str],
    by_lane: Mapping[str, MetricVectors],
    contaminated: set[str],
) -> list[ItemOutcome]:
    return [
        {
            "item_id": item_id,
            "question_type": question_types.get(item_id),
            "contaminated": item_id in contaminated,
            "lanes": {
                label: {metric: vectors[metric][i] for metric in METRICS}
                for label, vectors in by_lane.items()
            },
        }
        for i, item_id in enumerate(item_ids)
    ]


def compare_context_strategies(
    lanes: Mapping[str, CaseRows],
    question_types: Mapping[str, str],
    *,
    baseline: str = LANE_CLOSED_BOOK,
    run_dirs: Mapping[str, list[str]] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> ContextAblationReport:
    """Compare every context lane against `baseline`, sliced by question type."""
    if baseline not in lanes:
        raise ValueError(f"baseline lane {baseline!r} is not among the scored lanes")
    item_ids = shared_item_ids(lanes)
    by_lane = {label: lane_vectors(rows, item_ids, METRICS) for label, rows in lanes.items()}
    base_vectors = by_lane[baseline]
    skipped_by_lane = {label: skipped_item_ids(rows) for label, rows in lanes.items()}
    skipped = {item_id for ids in skipped_by_lane.values() for item_id in ids}
    grouped = slice_indexes([question_types.get(item_id) for item_id in item_ids])
    all_indexes = list(range(len(item_ids)))
    index_sets = bootstrap_index_sets(len(item_ids), resamples, seed)
    per_slice_sets = slice_index_sets(grouped, resamples, seed)
    lane_reports: dict[str, LaneReport] = {
        label: {
            "label": label,
            "run_dirs": list((run_dirs or {}).get(label, [])),
            "overall": slice_report(
                vectors, base_vectors, all_indexes, index_sets, confidence, METRICS
            ),
            "slices": {
                name: slice_report(
                    vectors, base_vectors, positions, per_slice_sets[name], confidence, METRICS
                )
                for name, positions in sorted(grouped.items())
            },
            "skipped_item_ids": skipped_by_lane[label],
        }
        for label, vectors in by_lane.items()
    }
    derived = _derived(by_lane, item_ids, skipped, index_sets, confidence, resamples, seed)
    contamination = contamination_report(baseline, lanes[baseline], item_ids)
    return {
        "n": len(item_ids),
        "baseline": baseline,
        "metrics": list(METRICS),
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "item_ids": item_ids,
        "lanes": lane_reports,
        "derived": derived,
        "contamination": contamination,
        "items": _items(item_ids, question_types, by_lane, set(contamination["item_ids"])),
        "verdict": decide(lane_reports, derived, contamination, baseline=baseline, n=len(item_ids)),
    }


__all__ = ["compare_context_strategies"]
