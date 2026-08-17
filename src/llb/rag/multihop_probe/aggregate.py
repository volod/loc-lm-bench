"""Aggregate item-level hop probes into bootstrap-backed slice reports."""

from collections.abc import Callable, Sequence

from llb.rag.fusion_evidence.slices import slice_indexes
from llb.rag.fusion_evidence.stats import bootstrap_index_sets, bootstrap_interval
from llb.rag.multihop_probe.diagnose import slice_diagnosis
from llb.rag.multihop_probe.models import (
    BudgetReport,
    EvidenceItem,
    HopOutcome,
    ItemProbe,
    MultiHopProbeReport,
    SliceProbe,
)


def _question_rank(hop: HopOutcome) -> int | None:
    return hop["question_rank"]


def _span_query_rank(hop: HopOutcome) -> int | None:
    return hop["span_query_rank"]


def _hop_hit_rate(
    probes: Sequence[ItemProbe], k: int, rank_of: Callable[[HopOutcome], int | None]
) -> float:
    """Share of labeled spans whose rank under one query form is within k."""
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


def assemble_probe_report(
    probes: Sequence[ItemProbe],
    items: Sequence[EvidenceItem],
    *,
    budgets: Sequence[int],
    depth: int,
    focus_slice: str,
    lane: str,
    resamples: int,
    confidence: float,
    seed: int,
) -> MultiHopProbeReport:
    """Build whole-set and question-type summaries from paired item probes."""
    grouped = slice_indexes([item.question_type for item in items], focus_slice)
    return {
        "lane": lane,
        "focus_slice": focus_slice,
        "budgets": list(budgets),
        "probe_depth": depth,
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "n_items": len(probes),
        "overall": _slice_probe(probes, budgets, depth, resamples, confidence, seed),
        "slices": {
            name: _slice_probe(
                [probes[position] for position in positions],
                budgets,
                depth,
                resamples,
                confidence,
                seed,
            )
            for name, positions in grouped.items()
        },
        "items": [probe for probe in probes if probe["question_type"] == focus_slice],
    }
