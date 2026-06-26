"""Compare retrieval backends on ONE gold set by the source-span metric (M6 residual 3).

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

from typing_extensions import TypedDict

from llb.contracts import ChunkRecord, RetrievalMetrics, SourceSpanRecord
from llb.rag.retrieval import evaluate_retrieval

# (question, gold source spans) -- the per-item input shared across every compared backend.
CompareItem = tuple[str, list[SourceSpanRecord]]


class Retriever(Protocol):
    """The RAG-store seam every compared backend implements (FAISS or GraphStore)."""

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


class ComparisonReport(TypedDict):
    """Per-backend span metrics over one goldset plus the recall winner (None if no backends)."""

    k: int
    n: int
    backends: dict[str, RetrievalMetrics]
    best_recall: str | None


def compare_retrieval(
    stores: dict[str, Retriever], items: list[CompareItem], k: int
) -> ComparisonReport:
    """Score each labeled backend's top-k retrieval over the same items; rank by recall@k."""
    per_backend: dict[str, RetrievalMetrics] = {}
    for label, store in stores.items():
        pairs = [(store.retrieve(question, k), spans) for question, spans in items]
        per_backend[label] = evaluate_retrieval(pairs, k)
    return {
        "k": k,
        "n": len(items),
        "backends": per_backend,
        "best_recall": _best_recall(per_backend),
    }


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
    return "\n".join(lines)


def load_compare_stores(config: Any) -> dict[str, Retriever]:
    """Load the three standard backends for `config`, skipping any whose store is not built.

    Returns `{faiss, graph/local_khop, graph/global_community}` -> store. A backend that has no
    built store on disk is skipped with a log line, so the comparison runs over whatever is present.
    """
    import logging

    from llb.executor.runner import _load_store
    from llb.graph.constants import (
        BACKEND_GRAPH,
        STRATEGY_GLOBAL_COMMUNITY,
        STRATEGY_LOCAL_KHOP,
    )

    log = logging.getLogger(__name__)
    plans = {
        "faiss": config.with_overrides(retrieval_backend="faiss"),
        f"{BACKEND_GRAPH}/{STRATEGY_LOCAL_KHOP}": config.with_overrides(
            retrieval_backend=BACKEND_GRAPH, retrieval_strategy=STRATEGY_LOCAL_KHOP
        ),
        f"{BACKEND_GRAPH}/{STRATEGY_GLOBAL_COMMUNITY}": config.with_overrides(
            retrieval_backend=BACKEND_GRAPH, retrieval_strategy=STRATEGY_GLOBAL_COMMUNITY
        ),
    }
    stores: dict[str, Retriever] = {}
    for label, plan in plans.items():
        try:
            stores[label] = _load_store(plan)
        except (FileNotFoundError, SystemExit) as exc:
            log.warning("[compare-retrieval] skip %s: not built (%s)", label, exc)
    return stores
