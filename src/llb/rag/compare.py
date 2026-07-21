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

from typing import Any, Protocol

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import ChunkRecord, RetrievalMetrics, SourceSpanRecord
from llb.rag.retrieval import evaluate_retrieval

# (question, gold source spans) -- the per-item input shared across every compared backend.
CompareItem = tuple[str, list[SourceSpanRecord]]

# Row labels of the hybrid comparison (hybrid-retrieval-uk).
ROW_DENSE = "dense"
ROW_HYBRID = "hybrid"
ROW_HYBRID_LEMMAS = "hybrid+lemmas"
ROW_ORACLE_DOC = "dense+oracle-doc"
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
    backends: dict[str, RetrievalMetrics]
    best_recall: str | None
    slices: NotRequired[dict[str, "ComparisonSlice"]]


class ComparisonSlice(TypedDict):
    n: int
    backends: dict[str, RetrievalMetrics]


def compare_retrieval(
    stores: dict[str, Retriever],
    items: list[CompareItem],
    k: int,
    slice_labels: list[str | None] | None = None,
) -> ComparisonReport:
    """Score each backend over the same items, with optional aligned question-type slices."""
    if slice_labels is not None and len(slice_labels) != len(items):
        raise ValueError("retrieval slice labels must align one-to-one with items")
    per_backend: dict[str, RetrievalMetrics] = {}
    pairs_by_backend: dict[str, list[Any]] = {}
    for label, store in stores.items():
        pairs = [(store.retrieve(question, k), spans) for question, spans in items]
        pairs_by_backend[label] = pairs
        per_backend[label] = evaluate_retrieval(pairs, k)
    report: ComparisonReport = {
        "k": k,
        "n": len(items),
        "backends": per_backend,
        "best_recall": _best_recall(per_backend),
    }
    if slice_labels is not None:
        labels = sorted({label for label in slice_labels if label} | {"comparative", "multi-hop"})
        report["slices"] = {
            slice_label: {
                "n": sum(label == slice_label for label in slice_labels),
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
    return report


def _best_recall(per_backend: dict[str, RetrievalMetrics]) -> str | None:
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


def format_comparison(report: ComparisonReport) -> str:
    """Render an ASCII comparison table (AGENTS.md: ASCII-only, no box-drawing)."""
    backends = report["backends"]
    lines = [f"[compare-retrieval] n={report['n']} k={report['k']}"]
    if not backends:
        lines.append("  (no backends loaded)")
        return "\n".join(lines)
    width = max(len(label) for label in backends)
    lines.append(f"  {'backend'.ljust(width)}   recall@k      mrr")
    for label in sorted(backends):
        metrics = backends[label]
        lines.append(
            f"  {label.ljust(width)}   {metrics['recall_at_k']:8.3f} {metrics['mrr']:8.3f}"
        )
    lines.append(f"  best (recall@k): {report['best_recall']}")
    for slice_label, slice_report in report.get("slices", {}).items():
        lines.append(f"  slice {slice_label} (n={slice_report['n']}):")
        for label in sorted(slice_report["backends"]):
            metrics = slice_report["backends"][label]
            lines.append(
                f"    {label.ljust(width)}   {metrics['recall_at_k']:8.3f} {metrics['mrr']:8.3f}"
            )
    return "\n".join(lines)
