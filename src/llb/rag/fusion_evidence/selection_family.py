"""The focus-slice row x metric family selected by the fusion verdict."""

from llb.rag.fusion_evidence.models import (
    FUSED_ROW_PREFIX,
    METRICS,
    ROUTED_ROW_PREFIX,
)
from llb.rag.fusion_evidence.randomization import seed_from_index_sets
from llb.rag.fusion_evidence.selection import (
    DEFAULT_SELECTION_RESAMPLES,
    SelectionAdjustment,
    selection_adjustment,
)
from llb.rag.fusion_evidence.slices import MetricVectors

_KEY_SEPARATOR = " :: "


def hypothesis_key(label: str, metric: str) -> str:
    """Stable machine-readable identity of one row x metric selection hypothesis."""
    return f"{label}{_KEY_SEPARATOR}{metric}"


def adjust_fusion_selection(
    vectors: dict[str, MetricVectors],
    *,
    baseline: str,
    indexes: list[int],
    resamples: int,
    index_sets: list[list[int]],
) -> SelectionAdjustment | None:
    """Declare and adjust the focus-slice family the verdict can select from."""
    if not index_sets or baseline not in vectors:
        return None
    candidates = {
        label: row
        for label, row in vectors.items()
        if label.startswith((FUSED_ROW_PREFIX, ROUTED_ROW_PREFIX)) and label != baseline
    }
    if not candidates:
        return None
    reference = vectors[baseline]
    family = {
        hypothesis_key(label, metric): [
            row[metric][index] - reference[metric][index] for index in indexes
        ]
        for label, row in sorted(candidates.items())
        for metric in METRICS
    }
    return selection_adjustment(
        family,
        resamples=max(resamples, DEFAULT_SELECTION_RESAMPLES),
        seed=seed_from_index_sets(index_sets),
    )
