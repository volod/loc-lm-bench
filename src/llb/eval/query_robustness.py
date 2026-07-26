"""Pure end-to-end query robustness evaluation and aggregation."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llb.eval.query_robustness_variants import generate_variant, resolve_variant_classes
from llb.goldset.schema import GoldItem
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, PairedComparison


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
    comparisons: dict[str, PairedComparison] = field(default_factory=dict)


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
    comparisons: dict[str, PairedComparison] = field(default_factory=dict)


@dataclass(frozen=True)
class RobustnessResult:
    rows: list[dict[str, Any]]
    clean_objective: float
    clean_recall: float
    lanes: tuple[LaneMetrics, ...]
    variant_classes: tuple[str, ...] = ()
    resamples: int = DEFAULT_RESAMPLES
    confidence: float = DEFAULT_CONFIDENCE


def summarize_query_robustness(
    rows: list[dict[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
    variant_classes: Sequence[str],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int,
) -> RobustnessResult:
    """Rebuild aggregates from persisted rows without executing a model."""
    from llb.eval.query_robustness_summary import summarize_query_robustness as summarize

    return summarize(
        rows,
        clean_rows,
        variant_classes,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
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
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> RobustnessResult:
    """Run every noisy class under every mitigation lane; clean rows stay external baseline rows."""
    classes = resolve_variant_classes(variant_classes)
    clean = {str(row["item_id"]): row for row in clean_rows}
    missing = [item.id for item in items if item.id not in clean]
    if missing:
        raise ValueError(f"clean baseline is missing item ids: {missing[:3]}")
    all_rows: list[dict[str, Any]] = []
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
    return summarize_query_robustness(
        all_rows,
        clean_rows,
        classes,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
