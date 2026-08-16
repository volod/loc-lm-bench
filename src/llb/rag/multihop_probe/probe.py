"""Measure why a two-hop item misses `all-spans@k`: the budget, the query, or the index.

Pure: the store is any object exposing `.retrieve(question, k)`, so the whole lane is unit-tested
with fake stores (no FAISS, no DuckDB, no GPU). Each item costs one retrieval per compared budget
(the curve, retrieved AT the budget an operator would serve), one deep retrieval (the per-hop
rank), and one retrieval per labeled span (the retrievability control).
"""

from collections.abc import Callable, Sequence

from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion_evidence.models import FOCUS_SLICE
from llb.rag.fusion_evidence.slices import slice_indexes
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
    bootstrap_interval,
)
from llb.rag.multihop_probe.diagnose import diagnose_item, item_min_budget, slice_diagnosis
from llb.rag.multihop_probe.models import (
    DEFAULT_BUDGETS,
    DEFAULT_PROBE_DEPTH,
    BudgetReport,
    EvidenceItem,
    HopOutcome,
    ItemBudgetOutcome,
    ItemProbe,
    MultiHopProbeReport,
    Retriever,
    SliceProbe,
    SourceSpanRecord,
)
from llb.rag.retrieval import (
    covered_span_count,
    first_hit_rank,
    recall_at_k,
    span_coverage_at_k,
)


def _hop(
    store: Retriever, span: SourceSpanRecord, index: int, deep: list[ChunkRecord], depth: int
) -> HopOutcome:
    """One labeled span ranked by the item's question and, as a control, by its own text."""
    own = store.retrieve(span["text"], depth)
    return {
        "span_index": index,
        "doc_id": span["doc_id"],
        "char_start": span["char_start"],
        "char_end": span["char_end"],
        "n_chars": span["char_end"] - span["char_start"],
        "question_rank": first_hit_rank(deep, [span]),
        "span_query_rank": first_hit_rank(own, [span]),
    }


def _item_budgets(
    store: Retriever, item: EvidenceItem, budgets: Sequence[int]
) -> list[ItemBudgetOutcome]:
    """The item's coverage curve, retrieved once per budget so each point is what k really buys."""
    outcomes: list[ItemBudgetOutcome] = []
    for k in budgets:
        hits = store.retrieve(item.question, k)
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
    store: Retriever, item: EvidenceItem, budgets: Sequence[int], depth: int
) -> ItemProbe:
    deep = store.retrieve(item.question, depth)
    hops = [_hop(store, span, index, deep, depth) for index, span in enumerate(item.spans)]
    ranks = [hop["question_rank"] for hop in hops]
    limiting = None if any(rank is None for rank in ranks) else max(rank or 0 for rank in ranks)
    return {
        "item_id": item.item_id,
        "question": item.question,
        "question_type": item.question_type,
        "n_spans": len(item.spans),
        "hops": hops,
        "budgets": _item_budgets(store, item, budgets),
        "limiting_rank": limiting,
        "min_budget": item_min_budget(limiting, budgets, depth),
        "diagnosis": diagnose_item(hops, budgets[0]),
    }


def _question_rank(hop: HopOutcome) -> int | None:
    return hop["question_rank"]


def _span_query_rank(hop: HopOutcome) -> int | None:
    return hop["span_query_rank"]


def _hop_hit_rate(
    probes: Sequence[ItemProbe], k: int, rank_of: Callable[[HopOutcome], int | None]
) -> float:
    """Share of LABELED SPANS (not items) whose rank under one query form is within k."""
    ranks = [rank_of(hop) for probe in probes for hop in probe["hops"]]
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _budget_report(
    probes: Sequence[ItemProbe],
    position: int,
    index_sets: list[list[int]],
    confidence: float,
) -> BudgetReport:
    """One budget's aggregate over one slice; the curve carries the interval, the rest are means."""
    outcomes = [probe["budgets"][position] for probe in probes]
    k = outcomes[0]["k"] if outcomes else 0
    all_spans = [outcome["all_spans_at_k"] for outcome in outcomes]
    return {
        "k": k,
        "n": len(probes),
        "all_spans_at_k": bootstrap_interval(all_spans, index_sets, confidence),
        "span_coverage": _mean([outcome["span_coverage"] for outcome in outcomes]),
        "recall_at_k": _mean([outcome["recall_at_k"] for outcome in outcomes]),
        "hop_hit_rate": _hop_hit_rate(probes, k, _question_rank),
        "span_query_hop_hit_rate": _hop_hit_rate(probes, k, _span_query_rank),
    }


def _slice_probe(
    probes: Sequence[ItemProbe],
    budgets: Sequence[int],
    depth: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> SliceProbe:
    index_sets = bootstrap_index_sets(len(probes), resamples, seed)
    return {
        "n": len(probes),
        "n_hops": sum(probe["n_spans"] for probe in probes),
        "curve": [
            _budget_report(probes, position, index_sets, confidence)
            for position in range(len(budgets))
        ],
        "diagnosis": slice_diagnosis(probes, budgets, depth),
    }


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
    ordered = tuple(sorted(dict.fromkeys(int(k) for k in budgets)))
    if not ordered:
        raise ValueError("at least one retrieval budget is required")
    depth = max(probe_depth, ordered[-1])
    probes = [_probe_item(store, item, ordered, depth) for item in items]
    grouped = slice_indexes([item.question_type for item in items], focus_slice)
    return {
        "lane": lane,
        "focus_slice": focus_slice,
        "budgets": list(ordered),
        "probe_depth": depth,
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "n_items": len(probes),
        "overall": _slice_probe(probes, ordered, depth, resamples, confidence, seed),
        "slices": {
            name: _slice_probe(
                [probes[position] for position in positions],
                ordered,
                depth,
                resamples,
                confidence,
                seed,
            )
            for name, positions in grouped.items()
        },
        "items": [probe for probe in probes if probe["question_type"] == focus_slice],
    }
