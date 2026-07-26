"""Aggregation and paired uncertainty over persisted query-robustness case rows."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from llb.eval.query_robustness import (
    LANE_OFF,
    MITIGATION_LANES,
    LaneMetrics,
    RobustnessResult,
    SubsetMetrics,
)
from llb.eval.query_robustness_uncertainty import delta_comparisons, recovery_comparisons
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    bootstrap_index_sets,
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _changed_metrics(
    rows: list[dict[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> SubsetMetrics:
    changed = [row for row in rows if bool(row.get("variant_changed", True))]
    objective = _mean([float(row["objective_score"]) for row in changed])
    recall = _mean([float(row["retrieval_hit"]) for row in changed])
    baselines = [dict(clean[str(row["item_id"])]) for row in changed]
    return SubsetMetrics(
        n=len(changed),
        objective_score=objective,
        recall_at_k=recall,
        objective_delta=objective - _mean([float(row["objective_score"]) for row in baselines]),
        recall_delta=recall - _mean([float(row["retrieval_hit"]) for row in baselines]),
        comparisons=delta_comparisons(
            changed,
            baselines,
            bootstrap_index_sets(len(changed), resamples, seed),
            confidence,
        ),
    )


def _lane_metrics(
    variant_class: str,
    mitigation: str,
    rows: list[dict[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    clean_objective: float,
    clean_recall: float,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> LaneMetrics:
    objective = _mean([float(row["objective_score"]) for row in rows])
    recall = _mean([float(row["retrieval_hit"]) for row in rows])
    shared = [
        row
        for row in rows
        if float(row["retrieval_hit"]) > 0
        and float(clean[str(row["item_id"])]["retrieval_hit"]) > 0
    ]
    baselines = [dict(clean[str(row["item_id"])]) for row in rows]
    return LaneMetrics(
        variant_class=variant_class,
        mitigation=mitigation,
        n=len(rows),
        errors=sum(str(row.get("status", "ok")) != "ok" for row in rows),
        objective_score=objective,
        recall_at_k=recall,
        objective_delta=objective - clean_objective,
        recall_delta=recall - clean_recall,
        shared_hit_n=len(shared),
        generation_delta_on_shared_hits=_mean(
            [
                float(row["objective_score"]) - float(clean[str(row["item_id"])]["objective_score"])
                for row in shared
            ]
        ),
        changed=_changed_metrics(
            rows,
            clean,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        ),
        comparisons=delta_comparisons(
            rows,
            baselines,
            bootstrap_index_sets(len(rows), resamples, seed),
            confidence,
        ),
    )


def _with_recovery(
    metric: LaneMetrics,
    raw: LaneMetrics,
    rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> LaneMetrics:
    """Credit a mitigation lane with what it restored against its class's unmitigated lane."""
    changed = [row for row in rows if bool(row.get("variant_changed", True))]
    raw_changed = [row for row in raw_rows if bool(row.get("variant_changed", True))]
    return replace(
        metric,
        objective_recovery=metric.objective_score - raw.objective_score,
        recall_recovery=metric.recall_at_k - raw.recall_at_k,
        changed=replace(
            metric.changed,
            objective_recovery=metric.changed.objective_score - raw.changed.objective_score,
            recall_recovery=metric.changed.recall_at_k - raw.changed.recall_at_k,
            comparisons={
                **metric.changed.comparisons,
                **recovery_comparisons(
                    changed,
                    raw_changed,
                    bootstrap_index_sets(len(changed), resamples, seed),
                    confidence,
                ),
            },
        ),
        comparisons={
            **metric.comparisons,
            **recovery_comparisons(
                rows,
                raw_rows,
                bootstrap_index_sets(len(rows), resamples, seed),
                confidence,
            ),
        },
    )


def _group_rows(
    rows: list[dict[str, Any]],
    item_ids: list[str],
    variant_class: str,
    mitigation: str,
) -> list[dict[str, Any]]:
    """Select and align one complete class/lane block to the clean item order."""
    selected = [
        row
        for row in rows
        if row["variant_class"] == variant_class and row["mitigation"] == mitigation
    ]
    by_id = {str(row["item_id"]): row for row in selected}
    if len(by_id) != len(selected) or set(by_id) != set(item_ids):
        raise ValueError(
            f"query robustness rows are incomplete or duplicated for {variant_class}/{mitigation}"
        )
    return [by_id[item_id] for item_id in item_ids]


def summarize_query_robustness(
    rows: list[dict[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
    variant_classes: Sequence[str],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int,
) -> RobustnessResult:
    """Rebuild every aggregate and its paired annotation from persisted per-case rows."""
    classes = tuple(variant_classes)
    clean = {str(row["item_id"]): row for row in clean_rows}
    item_ids = list(dict.fromkeys(str(row["item_id"]) for row in rows))
    missing = [item_id for item_id in item_ids if item_id not in clean]
    if missing:
        raise ValueError(f"clean baseline is missing item ids: {missing[:3]}")
    clean_objective = _mean([float(clean[item_id]["objective_score"]) for item_id in item_ids])
    clean_recall = _mean([float(clean[item_id]["retrieval_hit"]) for item_id in item_ids])
    grouped = {
        (variant_class, lane.id): _group_rows(rows, item_ids, variant_class, lane.id)
        for variant_class in classes
        for lane in MITIGATION_LANES
    }
    metrics = [
        _lane_metrics(
            variant_class,
            lane.id,
            grouped[(variant_class, lane.id)],
            clean,
            clean_objective,
            clean_recall,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
        for variant_class in classes
        for lane in MITIGATION_LANES
    ]
    raw_by_class = {
        metric.variant_class: metric for metric in metrics if metric.mitigation == LANE_OFF.id
    }
    with_recovery = tuple(
        _with_recovery(
            metric,
            raw_by_class[metric.variant_class],
            grouped[(metric.variant_class, metric.mitigation)],
            grouped[(metric.variant_class, LANE_OFF.id)],
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
        if metric.mitigation != LANE_OFF.id
        else metric
        for metric in metrics
    )
    return RobustnessResult(
        rows,
        clean_objective,
        clean_recall,
        with_recovery,
        classes,
        resamples,
        confidence,
    )
