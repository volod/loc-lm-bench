"""Score two `run-eval` bundles against each other per question-type slice (pure).

Everything here is file-driven: the input is one list of canonical per-case rows per lane plus the
question-type sidecar labels, so the whole comparison is unit-tested with dict rows -- no backend,
no store, no GPU. Uncertainty reuses the fusion-evidence paired bootstrap, because this lane asks
the same small-sample question about the same multi-hop slice and must be readable beside it. Item
alignment reuses `llb.eval.paired_cases`, shared with the context ablation.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from llb.eval.answer_quality.models import (
    ANSWER_COVERAGE_METRICS,
    BASE_METRICS,
    CONTEXT_METRICS,
    COVERAGE_METRICS,
    COVERAGE_PRIORITY,
    FOCUS_SLICE,
    STATUS_OK,
    AnswerQualityReport,
    CrossReading,
    ItemOutcome,
    LaneReport,
)
from llb.eval.answer_quality.verdict import decide
from llb.eval.paired_cases import CaseRows, lane_vectors, shared_item_ids
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


def resolve_metrics(lanes: Mapping[str, CaseRows]) -> tuple[str, ...]:
    """The base columns plus every sidecar column EVERY lane measured on EVERY case.

    A sidecar column present in only some lanes (or some cases) would compare a measured value
    against a missing one, so it is dropped rather than defaulted to zero.
    """
    extra = tuple(
        metric
        for metric in COVERAGE_METRICS + ANSWER_COVERAGE_METRICS + CONTEXT_METRICS
        if all(metric in row for rows in lanes.values() for row in rows)
    )
    return BASE_METRICS + extra


def coverage_metric(metrics: tuple[str, ...]) -> str:
    """The strongest coverage metric available for the retrieval-only verdict."""
    return next(metric for metric in COVERAGE_PRIORITY if metric in metrics)


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


@dataclass(frozen=True)
class _Draw:
    """Everything a paired reading needs once the per-item vectors exist.

    One draw is shared by the lane reports and by the cross readings, so a lane compared against
    the baseline and the same lane compared against its own smaller budget rest on the identical
    resample indexes (common random numbers) rather than on two independent draws.
    """

    grouped: dict[str, list[int]]
    all_indexes: list[int]
    index_sets: list[list[int]]
    per_slice_sets: dict[str, list[list[int]]]
    confidence: float
    metrics: tuple[str, ...]


def not_ok_cases(rows: CaseRows) -> int:
    """Cases this lane never answered -- a timeout scores zero exactly like a wrong answer."""
    return sum(1 for row in rows if str(row.get("status", STATUS_OK)) != STATUS_OK)


def _lane_report(
    label: str,
    vectors: LaneVectors,
    base: LaneVectors,
    draw: _Draw,
    run_dirs: list[str],
    not_ok: int = 0,
) -> LaneReport:
    """One row's overall and per-slice readings against `base`."""
    return {
        "label": label,
        "run_dirs": run_dirs,
        "not_ok": not_ok,
        "overall": slice_report(
            vectors, base, draw.all_indexes, draw.index_sets, draw.confidence, draw.metrics
        ),
        "slices": {
            name: slice_report(
                vectors, base, positions, draw.per_slice_sets[name], draw.confidence, draw.metrics
            )
            for name, positions in sorted(draw.grouped.items())
        },
    }


def _cross_readings(
    cross_baselines: Mapping[str, str],
    by_lane: Mapping[str, LaneVectors],
    run_dirs: Mapping[str, list[str]],
    draw: _Draw,
    not_ok: Mapping[str, int],
) -> dict[str, CrossReading]:
    """Each named lane read against another SCORED lane instead of against the report baseline."""
    readings: dict[str, CrossReading] = {}
    for label, base_label in cross_baselines.items():
        for name in (label, base_label):
            if name not in by_lane:
                raise ValueError(f"cross reading names lane {name!r}, which was not scored")
        report = _lane_report(
            label,
            by_lane[label],
            by_lane[base_label],
            draw,
            list(run_dirs.get(label, [])),
            not_ok.get(label, 0),
        )
        readings[label] = {**report, "base_lane": base_label}
    return readings


def compare_answer_quality(
    lanes: Mapping[str, CaseRows],
    question_types: Mapping[str, str],
    *,
    baseline: str,
    run_dirs: Mapping[str, list[str]] | None = None,
    focus_slice: str = FOCUS_SLICE,
    cross_baselines: Mapping[str, str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> AnswerQualityReport:
    """Compare every lane's per-case scores against `baseline`, sliced by question type.

    `cross_baselines` names extra readings that are NOT against the report baseline -- a budget
    sweep pairs each raised-budget cell against the same row's shipped-budget cell, which the
    single-baseline table cannot express.
    """
    if baseline not in lanes:
        raise ValueError(f"baseline lane {baseline!r} is not among the scored lanes")
    item_ids = shared_item_ids(lanes)
    metrics = resolve_metrics(lanes)
    by_lane = {label: lane_vectors(rows, item_ids, metrics) for label, rows in lanes.items()}
    grouped = slice_indexes([question_types.get(item_id) for item_id in item_ids], focus_slice)
    draw = _Draw(
        grouped=grouped,
        all_indexes=list(range(len(item_ids))),
        index_sets=bootstrap_index_sets(len(item_ids), resamples, seed),
        per_slice_sets=slice_index_sets(grouped, resamples, seed),
        confidence=confidence,
        metrics=metrics,
    )
    dirs = dict(run_dirs or {})
    lane_reports: dict[str, LaneReport] = {
        label: _lane_report(
            label,
            vectors,
            by_lane[baseline],
            draw,
            list(dirs.get(label, [])),
            not_ok_cases(lanes[label]),
        )
        for label, vectors in by_lane.items()
    }
    report: AnswerQualityReport = {
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
            confidence=confidence,
        ),
    }
    if cross_baselines:
        report["cross_readings"] = _cross_readings(
            cross_baselines,
            by_lane,
            dirs,
            draw,
            {label: lane_reports[label]["not_ok"] for label in lane_reports},
        )
    return report


__all__ = ["compare_answer_quality", "coverage_metric", "resolve_metrics"]
