"""Candidate x adoption-bar family selected by the embedder bake-off verdict."""

from collections.abc import Sequence

from llb.rag.embedding_bakeoff_uncertainty import (
    BARS,
    METRIC_RECALL,
    MetricVectors,
)
from llb.rag.fusion_evidence.randomization import seed_from_index_sets
from llb.rag.fusion_evidence.selection import (
    DEFAULT_SELECTION_RESAMPLES,
    SelectionAdjustment,
    selection_adjustment,
)
from llb.rag.fusion_evidence.stats import bootstrap_index_sets

_KEY_SEPARATOR = " :: "


def hypothesis_key(model: str, bar: str) -> str:
    return f"{model}{_KEY_SEPARATOR}{bar}"


def _deltas(candidate: MetricVectors, reference: MetricVectors, bar: str) -> list[float]:
    if len(candidate[bar]) != len(reference[bar]):
        raise ValueError("selection candidates must use the baseline's aligned items")
    return [value - base for value, base in zip(candidate[bar], reference[bar])]


def adjust_bakeoff_selection(
    vectors: dict[str, MetricVectors],
    baseline: str | None,
    bars: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> SelectionAdjustment | None:
    """Adjust the candidate x enabled-bar family from which an adoption may be selected."""
    if baseline is None or baseline not in vectors or resamples <= 0:
        return None
    reference = vectors[baseline]
    family = {
        hypothesis_key(model, bar): _deltas(candidate, reference, bar)
        for model, candidate in sorted(vectors.items())
        if model != baseline
        for bar in BARS
        if bar in bars
    }
    if not family:
        return None
    index_sets = bootstrap_index_sets(len(reference[METRIC_RECALL]), resamples, seed)
    return selection_adjustment(
        family,
        resamples=max(resamples, DEFAULT_SELECTION_RESAMPLES),
        seed=seed_from_index_sets(index_sets),
    )


def selection_note(adjustment: SelectionAdjustment | None) -> str:
    if adjustment is None:
        return ""
    detail = ", ".join(
        f"{key} raw p={entry['unadjusted_p']:.4f}, adjusted p={entry['adjusted_p']:.4f}"
        for key, entry in adjustment["p_values"].items()
    )
    return (
        f"; selection adjustment ({adjustment['family_size']} candidate-bar hypotheses, "
        f"Westfall-Young step-down max-T): {detail}"
    )
