"""Recovery annotations for query-robustness mitigation lanes."""

from dataclasses import replace
from typing import Any

from llb.eval.query_robustness import LANE_OFF, LaneMetrics
from llb.eval.query_robustness_uncertainty import recovery_comparisons
from llb.rag.fusion_evidence.stats import bootstrap_index_sets


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
    """Credit a mitigation lane with what it restored against its unmitigated lane."""
    changed = [row for row in rows if bool(row.get("variant_changed", True))]
    raw_changed = [row for row in raw_rows if bool(row.get("variant_changed", True))]
    return replace(
        metric,
        objective_recovery=metric.objective_score - raw.objective_score,
        recall_recovery=metric.recall_at_k - raw.recall_at_k,
        mrr_recovery=metric.mrr - raw.mrr,
        changed=replace(
            metric.changed,
            objective_recovery=metric.changed.objective_score - raw.changed.objective_score,
            recall_recovery=metric.changed.recall_at_k - raw.changed.recall_at_k,
            mrr_recovery=metric.changed.mrr - raw.changed.mrr,
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


def add_recovery(
    metrics: list[LaneMetrics],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[LaneMetrics, ...]:
    """Apply recovery comparisons to every mitigated lane."""
    raw_by_class = {
        metric.variant_class: metric for metric in metrics if metric.mitigation == LANE_OFF.id
    }
    recovered: list[LaneMetrics] = []
    for metric in metrics:
        if metric.mitigation == LANE_OFF.id:
            recovered.append(metric)
            continue
        recovered.append(
            _with_recovery(
                metric,
                raw_by_class[metric.variant_class],
                grouped[(metric.variant_class, metric.mitigation)],
                grouped[(metric.variant_class, LANE_OFF.id)],
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
        )
    return tuple(recovered)
