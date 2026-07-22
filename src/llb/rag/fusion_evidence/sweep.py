"""Score every compared retrieval row with small-sample uncertainty and a focus-slice verdict.

Pure: rows are any object exposing `.retrieve(question, k)`, so the whole lane is unit-tested with
fake stores (no FAISS, no DuckDB, no GPU). Each row is retrieved ONCE per item; every metric,
slice, bootstrap interval, and paired delta is computed from those cached per-item vectors, so
adding a graph weight to the sweep costs one retrieval pass, not one per metric.
"""

from llb.rag.fusion_evidence.models import (
    FOCUS_SLICE,
    METRIC_ALL_SPANS,
    METRIC_COVERAGE,
    METRIC_MRR,
    METRIC_RECALL,
    METRICS,
    EvidenceItem,
    FusionEvidenceReport,
    ItemOutcome,
    Retriever,
    RowReport,
    SliceReport,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
    bootstrap_interval,
    paired_comparison,
)
from llb.rag.fusion_evidence.verdict import decide
from llb.rag.retrieval import all_spans_at_k, recall_at_k, reciprocal_rank, span_coverage_at_k

# label -> metric -> per-item values, in item order.
RowVectors = dict[str, list[float]]


def _item_vectors(store: Retriever, items: list[EvidenceItem], k: int) -> RowVectors:
    """Retrieve once per item and derive every per-item metric value from that one ranking."""
    vectors: RowVectors = {metric: [] for metric in METRICS}
    for item in items:
        hits = store.retrieve(item.question, k)
        vectors[METRIC_RECALL].append(recall_at_k(hits, item.spans, k))
        vectors[METRIC_ALL_SPANS].append(all_spans_at_k(hits, item.spans, k))
        vectors[METRIC_COVERAGE].append(span_coverage_at_k(hits, item.spans, k))
        vectors[METRIC_MRR].append(reciprocal_rank(hits[:k], item.spans))
    return vectors


def _slice_report(
    vectors: RowVectors,
    baseline: RowVectors,
    indexes: list[int],
    index_sets: list[list[int]],
    confidence: float,
) -> SliceReport:
    """Metrics + paired deltas for one row restricted to `indexes` (an item slice)."""
    picked = {metric: [vectors[metric][i] for i in indexes] for metric in METRICS}
    base = {metric: [baseline[metric][i] for i in indexes] for metric in METRICS}
    return {
        "n": len(indexes),
        "metrics": {
            metric: bootstrap_interval(picked[metric], index_sets, confidence) for metric in METRICS
        },
        "paired_vs_baseline": {
            metric: paired_comparison(picked[metric], base[metric], index_sets, confidence)
            for metric in METRICS
        },
    }


def _slice_indexes(items: list[EvidenceItem], focus_slice: str) -> dict[str, list[int]]:
    """Item positions per question type; the focus slice is always present, even when empty."""
    grouped: dict[str, list[int]] = {focus_slice: []}
    for position, item in enumerate(items):
        if item.question_type:
            grouped.setdefault(item.question_type, []).append(position)
    return grouped


def _focus_items(
    items: list[EvidenceItem], indexes: list[int], by_row: dict[str, RowVectors]
) -> list[ItemOutcome]:
    """The item-level paired ledger for the focus slice (small n -- readable per item)."""
    return [
        {
            "item_id": items[i].item_id,
            "question": items[i].question,
            "n_spans": len(items[i].spans),
            "rows": {
                label: {metric: vectors[metric][i] for metric in METRICS}
                for label, vectors in by_row.items()
            },
        }
        for i in indexes
    ]


def evaluate_fusion_evidence(
    stores: dict[str, Retriever],
    items: list[EvidenceItem],
    k: int,
    *,
    baseline: str,
    focus_slice: str = FOCUS_SLICE,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> FusionEvidenceReport:
    """Score every row overall and per question-type slice, then decide on the focus slice."""
    if baseline not in stores:
        raise ValueError(f"baseline row {baseline!r} is not among the compared rows")
    by_row = {label: _item_vectors(store, items, k) for label, store in stores.items()}
    base_vectors = by_row[baseline]
    index_sets = bootstrap_index_sets(len(items), resamples, seed)
    grouped = _slice_indexes(items, focus_slice)
    all_indexes = list(range(len(items)))
    # Draw each slice's resample sets ONCE and share them across every row: common random numbers
    # keep the rows comparable, and the draw cost stops scaling with the size of the weight grid.
    slice_index_sets = {
        name: bootstrap_index_sets(len(positions), resamples, seed)
        for name, positions in grouped.items()
    }
    rows: dict[str, RowReport] = {
        label: {
            "overall": _slice_report(vectors, base_vectors, all_indexes, index_sets, confidence),
            "slices": {
                name: _slice_report(
                    vectors, base_vectors, positions, slice_index_sets[name], confidence
                )
                for name, positions in sorted(grouped.items())
            },
        }
        for label, vectors in by_row.items()
    }
    return {
        "k": k,
        "n": len(items),
        "baseline": baseline,
        "focus_slice": focus_slice,
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "rows": rows,
        "focus_items": _focus_items(items, grouped[focus_slice], by_row),
        "verdict": decide(rows, baseline=baseline, focus_slice=focus_slice),
    }


__all__ = [
    "METRIC_ALL_SPANS",
    "METRIC_COVERAGE",
    "METRIC_MRR",
    "METRIC_RECALL",
    "evaluate_fusion_evidence",
]
