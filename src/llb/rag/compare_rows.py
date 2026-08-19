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


def duplicate_census(
    stores: dict[str, Retriever],
) -> tuple[dict[str, "DuplicateStats"], dict[str, str]]:
    """Duplicate stats per store, plus why the ones that KEPT their copies did not collapse.

    The second map rides along because the stats alone cannot say whether the copies were dropped:
    a store built with `--keep-duplicate-chunks`, or under a strategy whose vector is not a pure
    function of its text, MEASURES its repeats and indexes all of them.
    """
    from llb.rag.duplicates import kept_duplicates_reason

    census: dict[str, DuplicateStats] = {}
    kept: dict[str, str] = {}
    for label, store in stores.items():
        meta = getattr(store, "meta", None)
        if not isinstance(meta, dict):
            continue
        stats = meta.get("duplicates")
        if not isinstance(stats, dict):
            continue
        census[label] = cast("DuplicateStats", stats)
        reason = kept_duplicates_reason(meta, requested=not meta.get("collapse_duplicates", True))
        if reason is not None:
            kept[label] = reason
    return census, kept


def format_comparison(report: ComparisonReport) -> str:
    """Render an ASCII comparison report without coupling scoring to presentation."""
    from llb.rag.retrieval_comparison_report import format_comparison as render

    return render(report)
