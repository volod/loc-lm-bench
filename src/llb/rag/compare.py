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

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import (
    ChunkRecord,
    RetrievalMetrics,
    RetrievalPair,
    SourceSpanRecord,
)
from llb.rag.retrieval import evaluate_retrieval

if TYPE_CHECKING:  # `noise_floor` imports this module, so the type is a forward reference
    from llb.rag.duplicate_models import DuplicateStats
    from llb.rag.embedding_bakeoff_uncertainty import MetricVectors, PairedRow
    from llb.rag.noise_floor_models import NoiseFloorReport
    from llb.rag.retrieval_comparison_uncertainty import RetrievalComparisonVerdict

# (question, gold source spans) -- the per-item input shared across every compared backend.
CompareItem = tuple[str, list[SourceSpanRecord]]

# Row labels of the hybrid comparison (hybrid-retrieval-uk).
ROW_DENSE = "dense"
ROW_HYBRID = "hybrid"
ROW_HYBRID_LEMMAS = "hybrid+lemmas"
ROW_ORACLE_DOC = "dense+oracle-doc"
# BM25 alone: the hybrid store queried at fusion weight 0, which `weighted_rrf_fuse` resolves to
# an exact lexical passthrough. It is the row that reads a LEXICAL-side change (tokenizer,
# lemmatization, normalization) without the dense lane masking it inside the fusion.
ROW_LEXICAL = "lexical"
# Suffix of the reranked twin row (rerank-context-order): `<label>+rerank` scores the SAME
# store's candidates after the cross-encoder cut, so pre/post-rerank recall@k / MRR compare
# through the one `evaluate_retrieval` metric.
RERANK_ROW_SUFFIX = "+rerank"


class Retriever(Protocol):
    """The RAG-store seam every compared backend implements (FAISS or GraphStore)."""

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


class ComparisonReport(TypedDict):
    """Per-backend span metrics over one goldset plus the recall winner (None if no backends)."""

    k: int
    n: int
    backends: dict[str, "ComparisonLane"]
    best_recall: str | None
    paired_items: list["ComparisonItemOutcome"]
    uncertainty: "ComparisonUncertainty"
    verdict: "RetrievalComparisonVerdict"
    slices: NotRequired[dict[str, "ComparisonSlice"]]
    # Each lane's exact-duplicate census (`llb.rag.duplicates`), present when the compared
    # stores expose their build meta -- so a recall row is read next to how much of that
    # lane's index is repeated text, and whether the repeats are intra- or cross-document.
    duplicates: NotRequired[dict[str, "DuplicateStats"]]
    # Measurement floor under numeric score noise; present only when it was asked for
    # (`compare-retrieval --noise-floor`). See `llb.rag.noise_floor`.
    noise_floor: NotRequired["NoiseFloorReport"]


class ComparisonSlice(TypedDict):
    n: int
    backends: dict[str, RetrievalMetrics]


class ComparisonLane(RetrievalMetrics):
    paired_vs_baseline: NotRequired["PairedRow"]


class ComparisonItemOutcome(TypedDict):
    item_id: str
    lanes: dict[str, dict[str, float]]


class ComparisonUncertainty(TypedDict):
    baseline: str | None
    eligible_lanes: list[str]
    resamples: int
    confidence: float
    seed: int


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
) -> dict[str, ComparisonSlice]:
    labels = sorted({label for label in slice_labels if label} | {"comparative", "multi-hop"})
    return {
        slice_label: {
            "n": slice_labels.count(slice_label),
            "backends": {
                backend: evaluate_retrieval(
                    [pair for pair, label in zip(pairs, slice_labels) if label == slice_label],
                    k,
                )
                for backend, pairs in pairs_by_backend.items()
            },
        }
        for slice_label in labels
    }


@dataclass(frozen=True, slots=True)
class _ComparisonSettings:
    """The resolved reading contract for one comparison: baseline, eligible lanes, resampling.

    Resolved once, before anything is scored, because every defaulting and alignment rule here is
    about what the report is ALLOWED to claim -- and a rule applied halfway through scoring is a
    rule the report cannot state.
    """

    baseline: str | None
    eligible: list[str]
    resamples: int
    confidence: float
    seed: int

    @classmethod
    def resolve(
        cls,
        stores: dict[str, Retriever],
        items: list[CompareItem],
        *,
        slice_labels: list[str | None] | None,
        item_ids: Sequence[str] | None,
        baseline: str | None,
        eligible_lanes: Sequence[str] | None,
        resamples: int | None,
        confidence: float | None,
        seed: int | None,
    ) -> "_ComparisonSettings":
        """Refuse a misaligned or unscorable request, then fill every unstated knob."""
        from llb.rag.fusion_evidence.stats import (
            DEFAULT_CONFIDENCE,
            DEFAULT_RESAMPLES,
            DEFAULT_SEED,
        )

        _check_alignment(items, slice_labels, item_ids)
        resolved_baseline, eligible = _resolved_lanes(stores, baseline, eligible_lanes)
        return cls(
            baseline=resolved_baseline,
            eligible=eligible,
            resamples=DEFAULT_RESAMPLES if resamples is None else resamples,
            confidence=DEFAULT_CONFIDENCE if confidence is None else confidence,
            seed=DEFAULT_SEED if seed is None else seed,
        )


def _check_alignment(
    items: list[CompareItem],
    slice_labels: list[str | None] | None,
    item_ids: Sequence[str] | None,
) -> None:
    """Per-item inputs must line up with the items, or a paired row means nothing."""
    if slice_labels is not None and len(slice_labels) != len(items):
        raise ValueError("retrieval slice labels must align one-to-one with items")
    if item_ids is not None and len(item_ids) != len(items):
        raise ValueError("retrieval item ids must align one-to-one with items")


def _resolved_lanes(
    stores: dict[str, Retriever], baseline: str | None, eligible_lanes: Sequence[str] | None
) -> tuple[str | None, list[str]]:
    """Which lane everything is read against, and which lanes may win -- both must be scored."""
    if baseline is not None and baseline not in stores:
        raise ValueError(f"retrieval baseline lane `{baseline}` was not scored")
    resolved = baseline if baseline is not None else next(iter(stores), None)
    eligible = list(eligible_lanes) if eligible_lanes is not None else list(stores)
    unknown = [lane for lane in eligible if lane not in stores]
    if unknown:
        raise ValueError(f"unknown retrieval verdict lane(s): {', '.join(unknown)}")
    if resolved is not None and resolved not in eligible:
        eligible.insert(0, resolved)
    return resolved, eligible


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
    from llb.rag.embedding_bakeoff_uncertainty import item_vectors, paired_rows
    from llb.rag.retrieval_comparison_uncertainty import decide_verdict, selection_adjustment

    settings = _ComparisonSettings.resolve(
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
        report["slices"] = _slice_reports(pairs_by_backend, slice_labels, k)
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


def add_rerank_rows(
    stores: dict[str, Retriever], scorer: Any, candidates: int
) -> dict[str, Retriever]:
    """Add a `<label>+rerank` twin per compared store (rerank-context-order).

    Each twin wraps the SAME store in the cross-encoder rerank stage (retrieve `candidates`,
    rerank, keep k), so the report shows the pre/post-rerank recall@k / MRR delta per backend.
    The oracle-doc headroom row is skipped (it is a diagnostic bound, not a rankable config).
    `scorer` is the injectable `RerankScorer` (a fake in tests; `CrossEncoderReranker` real).
    """
    from llb.rag.rerank import RerankingRetriever

    out: dict[str, Retriever] = dict(stores)
    for label, store in stores.items():
        if label == ROW_ORACLE_DOC:
            continue
        out[f"{label}{RERANK_ROW_SUFFIX}"] = RerankingRetriever(
            store, scorer, candidates=candidates
        )
    return out


def duplicate_census(stores: dict[str, Retriever]) -> dict[str, "DuplicateStats"]:
    """Each store's measured duplicate stats, for the stores that carry build meta.

    A graph or fake store has no `meta['duplicates']`, so it simply contributes no row: the census
    is an additive reading of the lanes that were built by `RagStore.build`.
    """
    census: dict[str, DuplicateStats] = {}
    for label, store in stores.items():
        meta = getattr(store, "meta", None)
        stats = meta.get("duplicates") if isinstance(meta, dict) else None
        if isinstance(stats, dict):
            census[label] = cast("DuplicateStats", stats)
    return census


def format_comparison(report: ComparisonReport) -> str:
    """Render an ASCII comparison report without coupling scoring to its presentation."""
    from llb.rag.retrieval_comparison_report import format_comparison as render

    return render(report)
