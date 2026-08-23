"""Optional rerank/stitch/duplicate rows and report rendering for retrieval comparisons."""

from typing import TYPE_CHECKING, Any, cast

from llb.rag.comparison.models import (
    RERANK_ROW_SUFFIX,
    ROW_ORACLE_DOC,
    STITCH_ROW_SUFFIX,
    ComparisonReport,
    Retriever,
    StitchLaneReport,
)

if TYPE_CHECKING:
    from llb.rag.duplicates.models import DuplicateStats

# Tolerance the invariance check reads a metric as reproduced at: stitching changes no retrieved
# character, so the two lanes' means are computed from identical per-item values and differ only
# by float summation order.
INVARIANCE_TOLERANCE = 1e-9


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


def add_stitch_rows(stores: dict[str, Retriever]) -> dict[str, Retriever]:
    """Add an assembly-time stitched twin for each non-oracle retrieval lane.

    The twin wraps its base lane, so it retrieves the same top-k and differs ONLY in how many
    blocks that evidence arrives in -- the paired reading of the lever against its own base.
    """
    from llb.rag.stitching import StitchingRetriever

    out: dict[str, Retriever] = dict(stores)
    for label, store in stores.items():
        if label != ROW_ORACLE_DOC:
            out[f"{label}{STITCH_ROW_SUFFIX}"] = StitchingRetriever(store)
    return out


def stitch_report(
    report: ComparisonReport, stores: dict[str, Retriever]
) -> dict[str, StitchLaneReport]:
    """Per stitched lane: what it merged, and whether it reproduced its base lane's finding metrics.

    The invariance is the reading's own precondition -- a stitched lane that moved recall or
    coverage did not reflow evidence, it changed it -- so it is recorded per lane rather than
    asserted in prose.
    """
    out: dict[str, StitchLaneReport] = {}
    for label, store in stores.items():
        census = getattr(store, "census", None)
        if not label.endswith(STITCH_ROW_SUFFIX) or not callable(census):
            continue
        base = label[: -len(STITCH_ROW_SUFFIX)]
        lanes = report["backends"]
        out[label] = StitchLaneReport(
            base=base,
            census=census(),
            recall_invariant=_reproduces(lanes, label, base, "recall_at_k"),
            coverage_invariant=_reproduces(lanes, label, base, "span_char_coverage_at_k"),
        )
    return out


def _reproduces(lanes: Any, label: str, base: str, metric: str) -> bool:
    """True when the stitched lane's metric matches its base lane's to float noise."""
    if base not in lanes or metric not in lanes[base] or metric not in lanes[label]:
        return False
    return abs(float(lanes[label][metric]) - float(lanes[base][metric])) <= INVARIANCE_TOLERANCE


def duplicate_census(
    stores: dict[str, Retriever],
) -> tuple[dict[str, "DuplicateStats"], dict[str, str]]:
    """Duplicate stats per store, plus why the ones that KEPT their copies did not collapse.

    The second map rides along because the stats alone cannot say whether the copies were dropped:
    a store built with `--keep-duplicate-chunks`, or under a strategy whose vector is not a pure
    function of its text, MEASURES its repeats and indexes all of them.
    """
    from llb.rag.duplicates.collapse import kept_duplicates_reason

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
    from llb.rag.comparison.report import format_comparison as render

    return render(report)
