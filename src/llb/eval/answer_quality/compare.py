"""Score two `run-eval` bundles against each other per question-type slice (pure).

Everything here is file-driven: the input is one list of canonical per-case rows per lane plus the
question-type sidecar labels, so the whole comparison is unit-tested with dict rows -- no backend,
no store, no GPU. Uncertainty reuses the fusion-evidence paired bootstrap, because this lane asks
the same small-sample question about the same multi-hop slice and must be readable beside it.
"""

from collections.abc import Mapping
from typing import Any

from llb.eval.answer_quality.models import (
    BASE_METRICS,
    COVERAGE_METRICS,
    COVERAGE_PRIORITY,
    FOCUS_SLICE,
    AnswerQualityReport,
    ItemOutcome,
    LaneReport,
)
from llb.eval.answer_quality.verdict import decide
from llb.rag.fusion_evidence.slices import (
    MetricVectors as LaneVectors,
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

CaseRows = list[Mapping[str, Any]]


def shared_item_ids(lanes: Mapping[str, CaseRows]) -> list[str]:
    """The sorted item ids every lane scored; raises when the lanes disagree.

    A paired comparison over different item sets is not a comparison, so a lane that dropped or
    added a case fails loudly instead of being silently intersected away.
    """
    if not lanes:
        raise ValueError("the comparison needs at least one scored lane")
    per_lane = {label: [str(row["item_id"]) for row in rows] for label, rows in lanes.items()}
    for label, ids in per_lane.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"lane {label!r} scored an item id more than once")
    reference_label, reference = next(iter(per_lane.items()))
    expected = set(reference)
    for label, ids in per_lane.items():
        if set(ids) != expected:
            missing = sorted(expected - set(ids))
            extra = sorted(set(ids) - expected)
            raise ValueError(
                f"lanes {reference_label!r} and {label!r} scored different item sets "
                f"(missing {missing[:3]}, extra {extra[:3]})"
            )
    return sorted(expected)


def resolve_metrics(lanes: Mapping[str, CaseRows]) -> tuple[str, ...]:
    """The base columns plus every coverage column EVERY lane measured on EVERY case.

    A coverage column present in only some lanes (or some cases) would compare a measured value
    against a missing one, so it is dropped rather than defaulted to zero.
    """
    extra = tuple(
        metric
        for metric in COVERAGE_METRICS
        if all(metric in row for rows in lanes.values() for row in rows)
    )
    return BASE_METRICS + extra


def coverage_metric(metrics: tuple[str, ...]) -> str:
    """The strongest coverage metric available for the retrieval-only verdict."""
    return next(metric for metric in COVERAGE_PRIORITY if metric in metrics)


def _lane_vectors(rows: CaseRows, item_ids: list[str], metrics: tuple[str, ...]) -> LaneVectors:
    """Per-metric values aligned to the shared item order."""
    by_id = {str(row["item_id"]): row for row in rows}
    return {
        metric: [float(by_id[item_id].get(metric, 0.0) or 0.0) for item_id in item_ids]
        for metric in metrics
    }


def _focus_items(
    item_ids: list[str],
    indexes: list[int],
    question_types: Mapping[str, str],
    by_lane: Mapping[str, LaneVectors],
    metrics: tuple[str, ...],
) -> list[ItemOutcome]:
    return [
        {
            "item_id": item_ids[i],
            "question_type": question_types.get(item_ids[i]),
            "lanes": {
                label: {metric: vectors[metric][i] for metric in metrics}
                for label, vectors in by_lane.items()
            },
        }
        for i in indexes
    ]


def compare_answer_quality(
    lanes: Mapping[str, CaseRows],
    question_types: Mapping[str, str],
    *,
    baseline: str,
    run_dirs: Mapping[str, list[str]] | None = None,
    focus_slice: str = FOCUS_SLICE,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> AnswerQualityReport:
    """Compare every lane's per-case scores against `baseline`, sliced by question type."""
    if baseline not in lanes:
        raise ValueError(f"baseline lane {baseline!r} is not among the scored lanes")
    item_ids = shared_item_ids(lanes)
    metrics = resolve_metrics(lanes)
    by_lane = {label: _lane_vectors(rows, item_ids, metrics) for label, rows in lanes.items()}
    base_vectors = by_lane[baseline]
    grouped = slice_indexes([question_types.get(item_id) for item_id in item_ids], focus_slice)
    all_indexes = list(range(len(item_ids)))
    index_sets = bootstrap_index_sets(len(item_ids), resamples, seed)
    per_slice_sets = slice_index_sets(grouped, resamples, seed)
    lane_reports: dict[str, LaneReport] = {
        label: {
            "label": label,
            "run_dirs": list((run_dirs or {}).get(label, [])),
            "overall": slice_report(
                vectors, base_vectors, all_indexes, index_sets, confidence, metrics
            ),
            "slices": {
                name: slice_report(
                    vectors, base_vectors, positions, per_slice_sets[name], confidence, metrics
                )
                for name, positions in sorted(grouped.items())
            },
        }
        for label, vectors in by_lane.items()
    }
    return {
        "n": len(item_ids),
        "baseline": baseline,
        "focus_slice": focus_slice,
        "metrics": list(metrics),
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "item_ids": item_ids,
        "lanes": lane_reports,
        "focus_items": _focus_items(
            item_ids, grouped[focus_slice], question_types, by_lane, metrics
        ),
        "verdict": decide(
            lane_reports,
            baseline=baseline,
            focus_slice=focus_slice,
            coverage=coverage_metric(metrics),
        ),
    }


__all__ = ["compare_answer_quality", "coverage_metric", "resolve_metrics", "shared_item_ids"]
