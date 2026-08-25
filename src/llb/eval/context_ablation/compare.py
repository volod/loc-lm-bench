"""Compare the context lanes over one identical item set (pure).

File-driven like every other evidence lane: the input is one list of canonical per-case rows per
lane plus the question-type sidecar labels, so the whole comparison is unit-tested with dict rows
-- no backend, no store, no GPU. The per-lane slices reuse the shared bootstrap middle layer, so
this artifact reads beside the retrieval sweep and the answer-quality comparison; what is new is
the derived-delta table and the contamination flag.

The comparison is assembled twice over: once over every scored item, and once per question type in
`per_slice.py`, from the same vectors and the same builders. The pooled pass is the corpus reading
and the sliced pass says which questions it was paid on.
"""

from collections.abc import Mapping, Sequence

from llb.eval.context_ablation.derived import (
    contamination_report,
    paired_deltas,
    skipped_item_ids,
)
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    METRICS,
    ContextAblationReport,
    ContextWindowBinding,
    ItemOutcome,
    LaneReport,
)
from llb.eval.context_ablation.per_slice import slice_readings
from llb.eval.context_ablation.verdict import decide
from llb.eval.context_ablation.verdict_adoption import decide_retrieved_document
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
    context_windows: Mapping[str, ContextWindowBinding | None] | None = None,
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
            "context_window": (context_windows or {}).get(label),
        }
        for label, vectors in by_lane.items()
    }
    derived = paired_deltas(
        by_lane, item_ids, skipped_by_lane, all_indexes, index_sets, confidence, resamples, seed
    )
    readings = slice_readings(
        grouped,
        per_slice_sets,
        by_lane=by_lane,
        item_ids=item_ids,
        baseline=baseline,
        baseline_rows=lanes[baseline],
        skipped_by_lane=skipped_by_lane,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    contamination = contamination_report(baseline, lanes[baseline], item_ids)
    verdict = decide(
        lane_reports,
        derived,
        contamination,
        baseline=baseline,
        n=len(item_ids),
        confidence=confidence,
    )
    # Two decisions, composed here rather than nested: the ablation reading ("what is retrieval
    # worth on this corpus") and the adoption call on the one lane an operator could ship.
    verdict["retrieved_document"] = decide_retrieved_document(
        derived, lane_reports, confidence=confidence
    )
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
        "slice_readings": readings,
        "contamination": contamination,
        "items": _items(item_ids, question_types, by_lane, set(contamination["item_ids"])),
        "verdict": verdict,
    }


__all__ = ["compare_context_strategies"]
