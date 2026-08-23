"""Compare retrieval backends on ONE gold set by the source-span metric (GraphRAG backend residual 3).

Quantifies when the GraphRAG multi-hop / narrative paths beat flat vector retrieval: it runs the
SAME goldset through several backends -- typically `{faiss, graph/local_khop, graph/global_community}`
-- and reports each one's `recall@k` / `MRR` (the model-independent retrieval axis the manifest's
backend + strategy already make comparable). Answer-quality comparison rides the normal
`run-eval --retrieval-backend ...` path (it needs a model); this tool isolates the retrieval signal.

Pure: it takes any object exposing `.retrieve(question, k) -> list[ChunkRecord]` (the RAG-store
seam), so it is unit-tested with fake stores -- no GPU, no FAISS, no DuckDB. Each backend reuses the
one `evaluate_retrieval` span metric, so graph and FAISS score on identical rules.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from llb.core.contracts.rag import (
    RetrievalMetrics,
    RetrievalPair,
)
from llb.rag.comparison.models import (
    FOCUS_SLICES,
    CompareItem,
    ComparisonItemOutcome,
    ComparisonLane,
    ComparisonReport,
    ComparisonSlice,
    Retriever,
)
from llb.rag.comparison.settings import ComparisonSettings
from llb.rag.retrieval import evaluate_retrieval

if TYPE_CHECKING:  # `noise_floor` imports this module, so the type is a forward reference
    from llb.rag.embedding_bakeoff.uncertainty import MetricVectors


def _retrieve_pairs(
    stores: dict[str, Retriever], items: list[CompareItem], k: int
) -> dict[str, list[RetrievalPair]]:
    return {
        label: [(store.retrieve(question, k), spans) for question, spans in items]
        for label, store in stores.items()
    }


def _slice_reports(
    pairs_by_backend: dict[str, list[Any]],
    slice_labels: list[str | None],
    k: int,
    settings: ComparisonSettings,
) -> dict[str, ComparisonSlice]:
    """Score every lane again on each question-type slice, paired against the baseline lane.

    The aggregate row cannot say which questions a change moved, and a slice's POINT estimate
    cannot say whether its movement is an item set -- a 14-item slice turns on one question. So a
    slice carries the same paired reading the aggregate does, drawn on that slice's own items from
    the SAME retrieval pass, and a lever is read on the slice it was aimed at with an interval.
    """
    labels = sorted({label for label in slice_labels if label} | set(FOCUS_SLICES))
    return {
        slice_label: _one_slice(pairs_by_backend, slice_labels, slice_label, k, settings)
        for slice_label in labels
    }


def _one_slice(
    pairs_by_backend: dict[str, list[Any]],
    slice_labels: list[str | None],
    slice_label: str,
    k: int,
    settings: ComparisonSettings,
) -> ComparisonSlice:
    """One slice's lane rows, plus its paired deltas when it labels an item and has a baseline."""
    from llb.rag.embedding_bakeoff.uncertainty import item_vectors, paired_rows

    sliced = {
        backend: [pair for pair, label in zip(pairs, slice_labels) if label == slice_label]
        for backend, pairs in pairs_by_backend.items()
    }
    n = slice_labels.count(slice_label)
    rows = {
        backend: cast(ComparisonLane, evaluate_retrieval(pairs, k))
        for backend, pairs in sliced.items()
    }
    if n and settings.baseline is not None:
        paired = paired_rows(
            {backend: item_vectors(pairs, k) for backend, pairs in sliced.items()},
            settings.baseline,
            resamples=settings.resamples,
            confidence=settings.confidence,
            seed=settings.seed,
        )
        for backend, row in paired.items():
            rows[backend]["paired_vs_baseline"] = row
    return {"n": n, "backends": rows}


def _lane_rows(
    pairs_by_backend: dict[str, list[Any]], paired: Mapping[str, Any], k: int
) -> dict[str, ComparisonLane]:
    """One scored row per backend, carrying its paired reading against the baseline when it has one."""
    rows: dict[str, ComparisonLane] = {}
    for label, pairs in pairs_by_backend.items():
        row = cast(ComparisonLane, evaluate_retrieval(pairs, k))
        if label in paired:
            row["paired_vs_baseline"] = paired[label]
        rows[label] = row
    return rows


def _paired_items(
    vectors: dict[str, "MetricVectors"], count: int, item_ids: Sequence[str] | None
) -> list[ComparisonItemOutcome]:
    """The per-item ledger a paired reading is recomputable from."""
    return [
        ComparisonItemOutcome(
            {
                "item_id": item_ids[index] if item_ids is not None else str(index),
                "lanes": {
                    lane: {metric: values[metric][index] for metric in values}
                    for lane, values in vectors.items()
                },
            }
        )
        for index in range(count)
    ]


def compare_retrieval(
    stores: dict[str, Retriever],
    items: list[CompareItem],
    k: int,
    slice_labels: list[str | None] | None = None,
    *,
    item_ids: Sequence[str] | None = None,
    baseline: str | None = None,
    eligible_lanes: Sequence[str] | None = None,
    resamples: int | None = None,
    confidence: float | None = None,
    seed: int | None = None,
) -> ComparisonReport:
    """Score each backend once and attach paired uncertainty against a named baseline lane."""
    from llb.rag.embedding_bakeoff.uncertainty import item_vectors, paired_rows
    from llb.rag.comparison.uncertainty import decide_verdict, selection_adjustment

    settings = ComparisonSettings.resolve(
        stores,
        items,
        slice_labels=slice_labels,
        item_ids=item_ids,
        baseline=baseline,
        eligible_lanes=eligible_lanes,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    pairs_by_backend = _retrieve_pairs(stores, items, k)
    vectors: dict[str, "MetricVectors"] = {
        label: item_vectors(pairs, k) for label, pairs in pairs_by_backend.items()
    }
    paired = (
        paired_rows(
            vectors,
            settings.baseline,
            resamples=settings.resamples,
            confidence=settings.confidence,
            seed=settings.seed,
        )
        if settings.baseline is not None
        else {}
    )
    per_backend = _lane_rows(pairs_by_backend, paired, k)
    best_eligible = _best_recall(
        {lane: per_backend[lane] for lane in settings.eligible if lane in per_backend}
    )
    report: ComparisonReport = {
        "k": k,
        "n": len(items),
        "backends": per_backend,
        "best_recall": _best_recall(per_backend),
        "paired_items": _paired_items(vectors, len(items), item_ids),
        "uncertainty": {
            "baseline": settings.baseline,
            "eligible_lanes": settings.eligible,
            "resamples": settings.resamples,
            "confidence": settings.confidence,
            "seed": settings.seed,
        },
        "verdict": decide_verdict(
            paired,
            baseline=settings.baseline,
            winner=best_eligible,
            confidence=settings.confidence,
            adjustment=selection_adjustment(
                vectors,
                settings.baseline,
                settings.eligible,
                resamples=settings.resamples,
                seed=settings.seed,
            ),
        ),
    }
    if slice_labels is not None:
        report["slices"] = _slice_reports(pairs_by_backend, slice_labels, k, settings)
    return report


def _best_recall(per_backend: Mapping[str, RetrievalMetrics]) -> str | None:
    """Label with the highest recall@k (tie-break: higher MRR, then label order)."""
    if not per_backend:
        return None
    return min(
        per_backend,
        key=lambda label: (
            -per_backend[label]["recall_at_k"],
            -per_backend[label]["mrr"],
            label,
        ),
    )
