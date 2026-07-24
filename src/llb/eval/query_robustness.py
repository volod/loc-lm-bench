"""Pure end-to-end query robustness evaluation and aggregation."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from llb.eval.query_robustness_variants import generate_variant, resolve_variant_classes
from llb.goldset.schema import GoldItem


@dataclass(frozen=True)
class MitigationLane:
    """One query-prep configuration every noise class is measured under.

    Splitting `normalize` from `normalize,typos` isolates the two mechanisms: normalization only
    inverts noise it can attribute (transliteration, homoglyphs, apostrophes), while the typos
    step additionally rewrites tokens to corpus surfaces, which is the step that carries
    vocabulary-correction risk. Reading them apart is what tells an operator whether a recovery
    came from safe normalization or from a correction they may not want on their corpus.
    """

    id: str
    steps: tuple[str, ...]
    typo_guard: bool

    @property
    def mitigated(self) -> bool:
        return bool(self.steps)


LANE_OFF = MitigationLane("off", (), False)
LANE_NORMALIZE = MitigationLane("normalize", ("normalize",), False)
LANE_NORMALIZE_TYPOS = MitigationLane("normalize,typos", ("normalize", "typos"), True)
MITIGATION_LANES: tuple[MitigationLane, ...] = (LANE_OFF, LANE_NORMALIZE, LANE_NORMALIZE_TYPOS)

QueryExecutor = Callable[[GoldItem, str, MitigationLane], Mapping[str, Any]]
Progress = Callable[[str], None]


@dataclass(frozen=True)
class SubsetMetrics:
    """Lane metrics restricted to the items a noise class actually perturbed.

    A single-mechanism class is a no-op on any question that carries none of its trigger
    characters -- `apostrophe_variant` cannot perturb a question without an apostrophe. Pooling
    those untouched items back into the lane mean drags every delta toward zero and makes a real
    effect on a handful of items unreadable, so the affected subset is measured separately
    against the SAME items' clean baseline.
    """

    n: int
    objective_score: float
    recall_at_k: float
    objective_delta: float
    recall_delta: float
    objective_recovery: float = 0.0
    recall_recovery: float = 0.0


@dataclass(frozen=True)
class LaneMetrics:
    variant_class: str
    mitigation: str
    n: int
    errors: int
    objective_score: float
    recall_at_k: float
    objective_delta: float
    recall_delta: float
    shared_hit_n: int
    generation_delta_on_shared_hits: float
    changed: SubsetMetrics
    objective_recovery: float = 0.0
    recall_recovery: float = 0.0


@dataclass(frozen=True)
class RobustnessResult:
    rows: list[dict[str, Any]]
    clean_objective: float
    clean_recall: float
    lanes: tuple[LaneMetrics, ...]
    variant_classes: tuple[str, ...] = ()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _changed_metrics(
    rows: list[dict[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
) -> SubsetMetrics:
    changed = [row for row in rows if bool(row.get("variant_changed", True))]
    objective = _mean([float(row["objective_score"]) for row in changed])
    recall = _mean([float(row["retrieval_hit"]) for row in changed])
    base_objective = _mean(
        [float(clean[str(row["item_id"])]["objective_score"]) for row in changed]
    )
    base_recall = _mean([float(clean[str(row["item_id"])]["retrieval_hit"]) for row in changed])
    return SubsetMetrics(
        n=len(changed),
        objective_score=objective,
        recall_at_k=recall,
        objective_delta=objective - base_objective,
        recall_delta=recall - base_recall,
    )


def _lane_metrics(
    variant_class: str,
    mitigation: str,
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
        mitigation=mitigation,
        n=len(rows),
        errors=sum(str(row.get("status", "ok")) != "ok" for row in rows),
        objective_score=objective,
        recall_at_k=recall,
        objective_delta=objective - clean_objective,
        recall_delta=recall - clean_recall,
        shared_hit_n=len(shared),
        generation_delta_on_shared_hits=generation_delta,
        changed=_changed_metrics(rows, clean),
    )


def _with_recovery(metric: LaneMetrics, raw: LaneMetrics) -> LaneMetrics:
    """Credit a mitigation lane with what it restored, against the `off` lane of its own class."""
    return replace(
        metric,
        objective_recovery=metric.objective_score - raw.objective_score,
        recall_recovery=metric.recall_at_k - raw.recall_at_k,
        changed=replace(
            metric.changed,
            objective_recovery=metric.changed.objective_score - raw.changed.objective_score,
            recall_recovery=metric.changed.recall_at_k - raw.changed.recall_at_k,
        ),
    )


def evaluate_query_robustness(
    items: list[GoldItem],
    clean_rows: Sequence[Mapping[str, Any]],
    execute: QueryExecutor,
    *,
    seed: int,
    typo_rate: float,
    variant_classes: Sequence[str] | None = None,
    progress: Progress | None = None,
) -> RobustnessResult:
    """Run every noisy class under every mitigation lane; clean rows stay external baseline rows."""
    classes = resolve_variant_classes(variant_classes)
    clean = {str(row["item_id"]): row for row in clean_rows}
    missing = [item.id for item in items if item.id not in clean]
    if missing:
        raise ValueError(f"clean baseline is missing item ids: {missing[:3]}")
    clean_objective = _mean([float(clean[item.id]["objective_score"]) for item in items])
    clean_recall = _mean([float(clean[item.id]["retrieval_hit"]) for item in items])
    all_rows: list[dict[str, Any]] = []
    metrics: list[LaneMetrics] = []
    total = len(items) * len(classes) * len(MITIGATION_LANES)
    completed = 0
    for variant_class in classes:
        for lane in MITIGATION_LANES:
            lane_rows: list[dict[str, Any]] = []
            for item in items:
                variant = generate_variant(
                    item.question,
                    variant_class,
                    item_id=item.id,
                    seed=seed,
                    typo_rate=typo_rate,
                )
                score = dict(execute(item, variant, lane))
                clean_row = clean[item.id]
                row = {
                    "probe": True,
                    "item_id": item.id,
                    "variant_class": variant_class,
                    "mitigation": lane.id,
                    "mitigated": lane.mitigated,
                    "mitigation_steps": list(lane.steps),
                    "mitigation_typo_guard": lane.typo_guard,
                    "seed": seed,
                    "typo_rate": typo_rate,
                    "clean_question": item.question,
                    "variant_question": variant,
                    "variant_changed": variant != item.question,
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
                    lane.id,
                    lane_rows,
                    clean,
                    clean_objective,
                    clean_recall,
                )
            )
    # Recovery is always read against the unmitigated lane of the SAME noise class, so each
    # mitigation lane's number answers "how much of this class's loss did this lane restore?".
    raw_by_class = {
        metric.variant_class: metric for metric in metrics if metric.mitigation == LANE_OFF.id
    }
    with_recovery = tuple(
        _with_recovery(metric, raw_by_class[metric.variant_class])
        if metric.mitigation != LANE_OFF.id
        else metric
        for metric in metrics
    )
    return RobustnessResult(all_rows, clean_objective, clean_recall, with_recovery, classes)
