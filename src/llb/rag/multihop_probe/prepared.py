"""Drive the per-hop probe through one reusable query-prep plan per item."""

from collections.abc import Sequence
from typing import Protocol

from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion_evidence.models import FOCUS_SLICE
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED
from llb.rag.multihop_probe.aggregate import assemble_probe_report
from llb.rag.multihop_probe.conversion import query_prep_conversion
from llb.rag.multihop_probe.models import (
    DEFAULT_BUDGETS,
    DEFAULT_PROBE_DEPTH,
    EvidenceItem,
    ItemProbe,
    MultiHopQueryPrepReport,
    Retriever,
)
from llb.rag.multihop_probe.probe import (
    QuestionRetrieve,
    _probe_item,
    probe_multihop_hops,
)
from llb.rag.query_prep.base import QueryPrepResult
from llb.rag.query_prep.retrieval import retrieve_prepared


class QueryPrepPipeline(Protocol):
    """The query-prep surface needed by the paired probe."""

    @property
    def steps(self) -> tuple[str, ...]: ...

    def process(self, query: str) -> QueryPrepResult: ...


def _prepared_retriever(store: Retriever, result: QueryPrepResult) -> QuestionRetrieve:
    def retrieve(k: int) -> list[ChunkRecord]:
        return retrieve_prepared(store, result, k)

    return retrieve


def compare_multihop_query_prep(
    store: Retriever,
    items: Sequence[EvidenceItem],
    query_prep: QueryPrepPipeline,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    probe_depth: int = DEFAULT_PROBE_DEPTH,
    focus_slice: str = FOCUS_SLICE,
    lane: str = "vector",
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> MultiHopQueryPrepReport:
    """Pair raw and prepared retrieval on the requested focus slice.

    Span-text ranks remain raw controls. Only the item question goes through query prep, and its
    generated plan is reused across every budget and the deep pass so one model response defines
    one paired item outcome. Non-focus items are intentionally not sent to the model: this lane
    measures a diagnosed cohort, rather than paying to transform unrelated question types.
    """
    focus_items = [item for item in items if item.question_type == focus_slice]
    if not focus_items:
        raise ValueError(f"query-prep probe focus slice is empty: {focus_slice}")
    baseline = probe_multihop_hops(
        store,
        focus_items,
        budgets=budgets,
        probe_depth=probe_depth,
        focus_slice=focus_slice,
        lane=lane,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    ordered = tuple(baseline["budgets"])
    depth = baseline["probe_depth"]
    baseline_items = {probe["item_id"]: probe for probe in baseline["items"]}
    prepared_probes: list[ItemProbe] = []
    for item in focus_items:
        result = query_prep.process(item.question)
        baseline_probe = baseline_items[item.item_id]
        controls = [hop["span_query_rank"] for hop in baseline_probe["hops"]]
        prepared_probes.append(
            _probe_item(
                item,
                ordered,
                depth,
                _prepared_retriever(store, result),
                controls,
                query_prep=result.provenance(),
            )
        )
    steps = list(query_prep.steps)
    prepared = assemble_probe_report(
        prepared_probes,
        focus_items,
        budgets=ordered,
        depth=depth,
        focus_slice=focus_slice,
        lane=f"{lane}+query-prep={','.join(steps)}",
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "query_prep_steps": steps,
        "baseline": baseline,
        "prepared": prepared,
        "conversion": query_prep_conversion(baseline, prepared),
    }
