"""Per-slice bootstrap reporting shared by every evidence lane that compares rows on a slice.

Two lanes ask the same statistical question about the same multi-hop slice -- the graph-weight
sweep (`fusion_evidence/sweep.py`, rows are retrievers) and the end-to-end answer comparison
(`llb/eval/answer_quality/`, rows are scored run bundles). What they share is everything AFTER the
per-item values exist: group items into question-type slices, draw one resample set per slice, and
turn each row's per-item vectors into `mean [lo, hi]` plus a paired delta against the baseline.

That middle layer lives here so the two lanes cannot drift into reporting the same slice two
slightly different ways. The metric names are a parameter, because the retrieval lane compares
fixed multi-span metrics while the answer lane's set depends on which columns its bundles carried.
"""

from collections.abc import Sequence

from typing_extensions import TypedDict

from llb.rag.fusion_evidence.stats import (
    Interval,
    PairedComparison,
    bootstrap_index_sets,
    bootstrap_interval,
    paired_comparison,
)

# metric -> per-item values, in the shared item order.
MetricVectors = dict[str, list[float]]


class SliceReport(TypedDict):
    """One row's metrics over one item slice, with its paired delta against the baseline row."""

    n: int
    metrics: dict[str, Interval]
    paired_vs_baseline: dict[str, PairedComparison]


def slice_indexes(
    question_types: Sequence[str | None], focus_slice: str | None = None
) -> dict[str, list[int]]:
    """Item positions per question type; a named focus slice is always present, even when empty.

    An empty focus slice must still appear, so a report over a set with no multi-hop item says
    `n=0` explicitly instead of omitting the slice the verdict is decided on. A lane with no focus
    slice (the context ablation decides on the whole set) passes None and gets only real types.
    """
    grouped: dict[str, list[int]] = {} if focus_slice is None else {focus_slice: []}
    for position, question_type in enumerate(question_types):
        if question_type:
            grouped.setdefault(question_type, []).append(position)
    return grouped


def slice_index_sets(
    grouped: dict[str, list[int]], resamples: int, seed: int
) -> dict[str, list[list[int]]]:
    """One resample draw per slice, to be SHARED by every compared row (common random numbers).

    Sharing keeps the rows comparable and stops the draw cost from scaling with the number of rows.
    """
    return {
        name: bootstrap_index_sets(len(positions), resamples, seed)
        for name, positions in grouped.items()
    }


def slice_report(
    vectors: MetricVectors,
    baseline: MetricVectors,
    indexes: list[int],
    index_sets: list[list[int]],
    confidence: float,
    metrics: Sequence[str],
) -> SliceReport:
    """Metrics + paired deltas for one row restricted to `indexes` (an item slice)."""
    picked = {metric: [vectors[metric][i] for i in indexes] for metric in metrics}
    base = {metric: [baseline[metric][i] for i in indexes] for metric in metrics}
    return {
        "n": len(indexes),
        "metrics": {
            metric: bootstrap_interval(picked[metric], index_sets, confidence) for metric in metrics
        },
        "paired_vs_baseline": {
            metric: paired_comparison(picked[metric], base[metric], index_sets, confidence)
            for metric in metrics
        },
    }
