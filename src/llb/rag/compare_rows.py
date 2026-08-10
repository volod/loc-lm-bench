"""Optional rerank/duplicate rows and report rendering for retrieval comparisons."""

from typing import TYPE_CHECKING, Any, cast

from llb.rag.compare_models import (
    RERANK_ROW_SUFFIX,
    ROW_ORACLE_DOC,
    ComparisonReport,
    Retriever,
)

if TYPE_CHECKING:
    from llb.rag.duplicate_models import DuplicateStats


def add_rerank_rows(
    stores: dict[str, Retriever], scorer: Any, candidates: int
) -> dict[str, Retriever]:
    """Add a reranked twin for each non-oracle retrieval lane."""
    from llb.rag.rerank import RerankingRetriever

    out: dict[str, Retriever] = dict(stores)
    for label, store in stores.items():
        if label != ROW_ORACLE_DOC:
            out[f"{label}{RERANK_ROW_SUFFIX}"] = RerankingRetriever(
                store, scorer, candidates=candidates
            )
    return out


def duplicate_census(stores: dict[str, Retriever]) -> dict[str, "DuplicateStats"]:
    """Read duplicate stats from stores that expose build metadata."""
    census: dict[str, DuplicateStats] = {}
    for label, store in stores.items():
        meta = getattr(store, "meta", None)
        stats = meta.get("duplicates") if isinstance(meta, dict) else None
        if isinstance(stats, dict):
            census[label] = cast("DuplicateStats", stats)
    return census


def format_comparison(report: ComparisonReport) -> str:
    """Render an ASCII comparison report without coupling scoring to presentation."""
    from llb.rag.retrieval_comparison_report import format_comparison as render

    return render(report)
