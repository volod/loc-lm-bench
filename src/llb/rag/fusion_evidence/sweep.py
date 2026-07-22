"""Score every compared retrieval row with small-sample uncertainty and a focus-slice verdict.

Pure: rows are any object exposing `.retrieve(question, k)`, so the whole lane is unit-tested with
fake stores (no FAISS, no DuckDB, no GPU). Each row is retrieved ONCE per item; every metric,
slice, bootstrap interval, and paired delta is computed from those cached per-item vectors, so
adding a graph weight to the sweep costs one retrieval pass, not one per metric.
"""

from llb.rag.fusion_evidence.models import (
    FOCUS_SLICE,
    AgreementReport,
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
)
from llb.rag.fusion_evidence.slices import (
    MetricVectors as RowVectors,
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
from llb.rag.fusion_evidence.verdict import decide
from llb.rag.retrieval import all_spans_at_k, recall_at_k, reciprocal_rank, span_coverage_at_k


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


def _agreement(store: Retriever, items: list[EvidenceItem], k: int) -> AgreementReport | None:
    """Cross-lane agreement of one row, when the row can measure it (only fused rows can).

    Read through an optional `lane_agreement(question, k)` seam rather than by isinstance, so the
    sweep stays a pure consumer of the `.retrieve` protocol and a fake row can report agreement.
    """
    measure = getattr(store, "lane_agreement", None)
    if not callable(measure) or not items:
        return None
    shared = [int(measure(item.question, k)) for item in items]
    with_shared = sum(1 for count in shared if count > 0)
    return {
        "questions": len(shared),
        "questions_with_shared_candidate": with_shared,
        "share_of_questions": with_shared / len(shared),
        "mean_shared_candidates": sum(shared) / len(shared),
    }


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
    grouped = slice_indexes([item.question_type for item in items], focus_slice)
    all_indexes = list(range(len(items)))
    per_slice_sets = slice_index_sets(grouped, resamples, seed)
    rows: dict[str, RowReport] = {}
    for label, vectors in by_row.items():
        row: RowReport = {
            "overall": slice_report(
                vectors, base_vectors, all_indexes, index_sets, confidence, METRICS
            ),
            "slices": {
                name: slice_report(
                    vectors, base_vectors, positions, per_slice_sets[name], confidence, METRICS
                )
                for name, positions in sorted(grouped.items())
            },
        }
        agreement = _agreement(stores[label], items, k)
        if agreement is not None:
            row["agreement"] = agreement
        rows[label] = row
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
