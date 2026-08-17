"""Measure why a two-hop item misses `all-spans@k`: the budget, the query, or the index.

Pure: the store is any object exposing `.retrieve(question, k)`, so the whole lane is unit-tested
with fake stores (no FAISS, no DuckDB, no GPU). Each item costs one retrieval per compared budget
(the curve, retrieved AT the budget an operator would serve), one deep retrieval (the per-hop
rank), and one retrieval per labeled span (the retrievability control).
"""

from collections.abc import Callable, Sequence

from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion_evidence.models import FOCUS_SLICE
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
)
from llb.rag.multihop_probe.aggregate import assemble_probe_report
from llb.rag.multihop_probe.diagnose import diagnose_item, item_min_budget
from llb.rag.multihop_probe.models import (
    DEFAULT_BUDGETS,
    DEFAULT_PROBE_DEPTH,
    EvidenceItem,
    HopOutcome,
    ItemBudgetOutcome,
    ItemProbe,
    MultiHopProbeReport,
    Retriever,
    SourceSpanRecord,
)
from llb.rag.retrieval import (
    covered_span_count,
    first_hit_rank,
    recall_at_k,
    span_coverage_at_k,
)


QuestionRetrieve = Callable[[int], list[ChunkRecord]]


def _raw_question_retriever(store: Retriever, question: str) -> QuestionRetrieve:
    def retrieve(k: int) -> list[ChunkRecord]:
        return store.retrieve(question, k)

    return retrieve


def _hop(
    span: SourceSpanRecord,
    index: int,
    deep: list[ChunkRecord],
    span_query_rank: int | None,
) -> HopOutcome:
    """One labeled span ranked by the item query plan and by its raw-text control."""
    return {
        "span_index": index,
        "doc_id": span["doc_id"],
        "char_start": span["char_start"],
        "char_end": span["char_end"],
        "n_chars": span["char_end"] - span["char_start"],
        "question_rank": first_hit_rank(deep, [span]),
        "span_query_rank": span_query_rank,
    }


def _span_query_ranks(
    store: Retriever, spans: Sequence[SourceSpanRecord], depth: int
) -> list[int | None]:
    """Raw span-text controls; query prep must never transform these favorable control queries."""
    return [first_hit_rank(store.retrieve(span["text"], depth), [span]) for span in spans]


def _item_budgets(
    item: EvidenceItem, budgets: Sequence[int], retrieve_question: QuestionRetrieve
) -> list[ItemBudgetOutcome]:
    """The item's coverage curve, retrieved once per budget so each point is what k really buys."""
    outcomes: list[ItemBudgetOutcome] = []
    for k in budgets:
        hits = retrieve_question(k)
        coverage = span_coverage_at_k(hits, item.spans, k)
        outcomes.append(
            {
                "k": k,
                "covered_spans": covered_span_count(hits, item.spans, k),
                "span_coverage": coverage,
                "all_spans_at_k": 1.0 if coverage == 1.0 else 0.0,
                "recall_at_k": recall_at_k(hits, item.spans, k),
            }
        )
    return outcomes


def _probe_item(
    item: EvidenceItem,
    budgets: Sequence[int],
    depth: int,
    retrieve_question: QuestionRetrieve,
    span_query_ranks: Sequence[int | None],
    *,
    query_prep: dict[str, object] | None = None,
) -> ItemProbe:
    deep = retrieve_question(depth)
    hops = [
        _hop(span, index, deep, span_query_ranks[index]) for index, span in enumerate(item.spans)
    ]
    ranks = [hop["question_rank"] for hop in hops]
    limiting = None if any(rank is None for rank in ranks) else max(rank or 0 for rank in ranks)
    probe: ItemProbe = {
        "item_id": item.item_id,
        "question": item.question,
        "question_type": item.question_type,
        "n_spans": len(item.spans),
        "hops": hops,
        "budgets": _item_budgets(item, budgets, retrieve_question),
        "limiting_rank": limiting,
        "min_budget": item_min_budget(limiting, budgets, depth),
        "diagnosis": diagnose_item(hops, budgets[0]),
    }
    if query_prep is not None:
        probe["query_prep"] = query_prep
    return probe


def _ordered_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    ordered = tuple(sorted(dict.fromkeys(int(k) for k in budgets)))
    if not ordered:
        raise ValueError("at least one retrieval budget is required")
    return ordered


def probe_multihop_hops(
    store: Retriever,
    items: Sequence[EvidenceItem],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    probe_depth: int = DEFAULT_PROBE_DEPTH,
    focus_slice: str = FOCUS_SLICE,
    lane: str = "vector",
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> MultiHopProbeReport:
    """Probe every item's per-hop retrievability and read the `all-spans@k` curve against it.

    `budgets` is the compared retrieval cutoff grid (ascending; its first entry is the operating
    budget the diagnosis is stated against). `probe_depth` is how deep a hop is searched for
    before the question counts as unable to reach it -- always at least the largest budget.
    """
    ordered = _ordered_budgets(budgets)
    depth = max(probe_depth, ordered[-1])
    if not any(item.question_type == focus_slice for item in items):
        raise ValueError(f"probe focus slice is empty: {focus_slice}")
    probes = []
    for item in items:
        retrieve_question = _raw_question_retriever(store, item.question)
        probes.append(
            _probe_item(
                item,
                ordered,
                depth,
                retrieve_question,
                _span_query_ranks(store, item.spans, depth),
            )
        )
    return assemble_probe_report(
        probes,
        items,
        budgets=ordered,
        depth=depth,
        focus_slice=focus_slice,
        lane=lane,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
