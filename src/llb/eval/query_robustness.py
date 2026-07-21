"""Pure end-to-end query robustness evaluation and aggregation."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llb.eval.query_robustness_variants import VARIANT_CLASSES, generate_variant
from llb.goldset.schema import GoldItem

MITIGATION_STEPS = ("normalize", "typos")
MITIGATION_TYPO_GUARD = True
QueryExecutor = Callable[[GoldItem, str, bool], Mapping[str, Any]]
Progress = Callable[[str], None]


@dataclass(frozen=True)
class LaneMetrics:
    variant_class: str
    mitigated: bool
    n: int
    errors: int
    objective_score: float
    recall_at_k: float
    objective_delta: float
    recall_delta: float
    shared_hit_n: int
    generation_delta_on_shared_hits: float
    objective_recovery: float = 0.0
    recall_recovery: float = 0.0


@dataclass(frozen=True)
class RobustnessResult:
    rows: list[dict[str, Any]]
    clean_objective: float
    clean_recall: float
    lanes: tuple[LaneMetrics, ...]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _lane_metrics(
    variant_class: str,
    mitigated: bool,
    rows: list[dict[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    clean_objective: float,
    clean_recall: float,
) -> LaneMetrics:
    objective = _mean([float(row["objective_score"]) for row in rows])
    recall = _mean([float(row["retrieval_hit"]) for row in rows])
    shared = [
        row
        for row in rows
        if float(row["retrieval_hit"]) > 0
        and float(clean[str(row["item_id"])]["retrieval_hit"]) > 0
    ]
    generation_delta = _mean(
        [
            float(row["objective_score"]) - float(clean[str(row["item_id"])]["objective_score"])
            for row in shared
        ]
    )
    return LaneMetrics(
        variant_class=variant_class,
        mitigated=mitigated,
        n=len(rows),
        errors=sum(str(row.get("status", "ok")) != "ok" for row in rows),
        objective_score=objective,
        recall_at_k=recall,
        objective_delta=objective - clean_objective,
        recall_delta=recall - clean_recall,
        shared_hit_n=len(shared),
        generation_delta_on_shared_hits=generation_delta,
    )


def evaluate_query_robustness(
    items: list[GoldItem],
    clean_rows: Sequence[Mapping[str, Any]],
    execute: QueryExecutor,
    *,
    seed: int,
    typo_rate: float,
    progress: Progress | None = None,
) -> RobustnessResult:
    """Run every noisy class with mitigation off/on; clean rows remain external baseline rows."""
    clean = {str(row["item_id"]): row for row in clean_rows}
    missing = [item.id for item in items if item.id not in clean]
    if missing:
        raise ValueError(f"clean baseline is missing item ids: {missing[:3]}")
    clean_objective = _mean([float(clean[item.id]["objective_score"]) for item in items])
    clean_recall = _mean([float(clean[item.id]["retrieval_hit"]) for item in items])
    all_rows: list[dict[str, Any]] = []
    metrics: list[LaneMetrics] = []
    total = len(items) * len(VARIANT_CLASSES) * 2
    completed = 0
    for variant_class in VARIANT_CLASSES:
        for mitigated in (False, True):
            lane_rows: list[dict[str, Any]] = []
            for item in items:
                variant = generate_variant(
                    item.question,
                    variant_class,
                    item_id=item.id,
                    seed=seed,
                    typo_rate=typo_rate,
                )
                score = dict(execute(item, variant, mitigated))
                clean_row = clean[item.id]
                row = {
                    "probe": True,
                    "item_id": item.id,
                    "variant_class": variant_class,
                    "mitigated": mitigated,
                    "mitigation_steps": list(MITIGATION_STEPS) if mitigated else [],
                    "mitigation_typo_guard": MITIGATION_TYPO_GUARD if mitigated else False,
                    "seed": seed,
                    "typo_rate": typo_rate,
                    "clean_question": item.question,
                    "variant_question": variant,
                    **score,
                    "clean_objective_score": float(clean_row["objective_score"]),
                    "clean_retrieval_hit": float(clean_row["retrieval_hit"]),
                }
                row["objective_delta"] = float(row["objective_score"]) - float(
                    clean_row["objective_score"]
                )
                row["recall_delta"] = float(row["retrieval_hit"]) - float(
                    clean_row["retrieval_hit"]
                )
                lane_rows.append(row)
                all_rows.append(row)
                completed += 1
                if progress is not None and (completed % 10 == 0 or completed == total):
                    progress(f"[query-robustness] completed {completed}/{total} variant cases")
            metrics.append(
                _lane_metrics(
                    variant_class,
                    mitigated,
                    lane_rows,
                    clean,
                    clean_objective,
                    clean_recall,
                )
            )
    raw_by_class = {metric.variant_class: metric for metric in metrics if not metric.mitigated}
    with_recovery = tuple(
        LaneMetrics(
            **{
                **metric.__dict__,
                "objective_recovery": metric.objective_score
                - raw_by_class[metric.variant_class].objective_score,
                "recall_recovery": metric.recall_at_k
                - raw_by_class[metric.variant_class].recall_at_k,
            }
        )
        if metric.mitigated
        else metric
        for metric in metrics
    )
    return RobustnessResult(all_rows, clean_objective, clean_recall, with_recovery)
