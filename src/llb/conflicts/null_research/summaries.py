"""Per-generation summary payloads for the independent-null research matrix.

Every generation returns the SAME envelope -- generation id, verdict, adopted methods, resolved
parameters, dataset geometries, rank baseline -- wrapped around the lanes only that generation
produces. The envelope and the settings shared by all four (`ResearchBudget`) are defined once
here, so each builder below reads as just the part where its generation genuinely differs: which
controls it needs, and which lanes it runs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llb.conflicts.constants import (
    RESEARCH_GENERATION_FOURTH,
    RESEARCH_GENERATION_INITIAL,
    RESEARCH_GENERATION_NEXT,
    RESEARCH_GENERATION_THIRD,
)
from llb.conflicts.null_research.generations.fourth import run_fourth_generation_candidates
from llb.conflicts.null_research.geometry import (
    CorpusGeometry,
    geometry_payload,
    prepare_geometry,
)
from llb.conflicts.null_research.generations.initial import run_initial_candidates
from llb.conflicts.null_research.generations.nextgen import run_next_generation_candidates
from llb.conflicts.null_research.generations.third import run_third_generation_candidates
from llb.conflicts.store_access import StoreView
from llb.core.contracts.common import JsonObject

if TYPE_CHECKING:
    from llb.prep.frontier.telemetry import LLMComplete
    from llb.rag.rerank import RerankScorer

EmbedTexts = Callable[[list[str]], object]

__all__ = [
    "EmbedTexts",
    "ResearchBudget",
    "fourth_generation_summary",
    "initial_summary",
    "next_generation_summary",
    "third_generation_summary",
]


@dataclass(frozen=True)
class ResearchBudget:
    """The settings every generation resolves against, whichever lanes it happens to run."""

    fpr: float
    rank_budget: int
    transfer_threshold: float
    max_goods_candidates: int
    seed: int

    def parameters(self, **generation_specific: object) -> JsonObject:
        """The recorded parameter block: the shared settings, then this generation's own."""
        return {
            "fpr": self.fpr,
            "rank_budget": self.rank_budget,
            "transfer_threshold": self.transfer_threshold,
            "max_goods_candidates": self.max_goods_candidates,
            **generation_specific,
            "seed": self.seed,
        }


def _adopted_methods(methods: list[JsonObject]) -> list[str]:
    return [str(method["method"]) for method in methods if method["gates"]["accepted"]]


def _lane_adopted(lanes: JsonObject, single_method_lane: str) -> list[str]:
    """Adopted names from the shared `methods` list plus one lane that reports a single method."""
    methods = lanes["methods"]
    single = lanes[single_method_lane]
    assert isinstance(methods, list) and isinstance(single, dict)
    adopted = _adopted_methods(methods)
    if single["gates"]["accepted"]:
        adopted.append(str(single["method"]))
    return adopted


def _datasets(corpora: dict[str, CorpusGeometry], **banks: CorpusGeometry) -> JsonObject:
    """Geometry payloads for the scored corpora plus whichever reference banks were collected."""
    return {
        **{name: geometry_payload(corpus) for name, corpus in corpora.items()},
        **{name: geometry_payload(bank) for name, bank in banks.items()},
    }


def _envelope(
    generation: str,
    adopted: list[str],
    parameters: JsonObject,
    datasets: JsonObject,
    rank: JsonObject,
    lanes: JsonObject,
) -> JsonObject:
    """The record every generation returns: what was adopted, under what settings, over what data."""
    return {
        "research_generation": generation,
        "verdict": "adopt" if adopted else "negative",
        "adopted_methods": adopted,
        "parameters": parameters,
        "datasets": datasets,
        "rank_baseline": rank,
        **lanes,
    }


def initial_summary(
    corpora: dict[str, CorpusGeometry],
    reference: CorpusGeometry,
    rank: JsonObject,
    embed: EmbedTexts,
    budget: ResearchBudget,
    *,
    permutations: int,
) -> JsonObject:
    """Cross-corpus, permutation, held-out, and labelled nulls against one general reference."""
    methods, repeats = run_initial_candidates(
        corpora,
        reference,
        rank,
        embed,
        fpr=budget.fpr,
        transfer_threshold=budget.transfer_threshold,
        max_goods_candidates=budget.max_goods_candidates,
        permutations=permutations,
        seed=budget.seed,
    )
    return _envelope(
        RESEARCH_GENERATION_INITIAL,
        _adopted_methods(methods),
        budget.parameters(minimum_permutations=permutations, resolved_permutations=repeats),
        _datasets(corpora, reference=reference),
        rank,
        {"methods": methods},
    )


def next_generation_summary(
    corpora: dict[str, CorpusGeometry],
    reference: CorpusGeometry,
    domain_reference: tuple[Path, StoreView],
    rank: JsonObject,
    embed: EmbedTexts,
    budget: ResearchBudget,
    *,
    matches_per_reference: int,
) -> JsonObject:
    """Matched, residual, cluster-FDR, and counterfactual nulls over two reference banks."""
    domain_geometry = prepare_geometry("domain_reference", *domain_reference)
    methods, traces = run_next_generation_candidates(
        corpora,
        {"general": reference, "domain": domain_geometry},
        rank,
        embed,
        fpr=budget.fpr,
        transfer_threshold=budget.transfer_threshold,
        max_goods_candidates=budget.max_goods_candidates,
        matches_per_reference=matches_per_reference,
        seed=budget.seed,
    )
    return _envelope(
        RESEARCH_GENERATION_NEXT,
        _adopted_methods(methods),
        budget.parameters(matches_per_reference=matches_per_reference),
        _datasets(corpora, general_reference=reference, domain_reference=domain_geometry),
        rank,
        {"methods": methods, "control_traces": traces},
    )


def third_generation_summary(
    corpora: dict[str, CorpusGeometry],
    reference: CorpusGeometry,
    domain_reference: tuple[Path, StoreView],
    rank: JsonObject,
    complete: "LLMComplete",
    budget: ResearchBudget,
    *,
    adjudicator_model: str,
    adjudication_budget: int,
    role_samples_per_type: int,
) -> JsonObject:
    """Model-adjudicated lanes: verified control roles plus the claim-tier precision curve."""
    domain_geometry = prepare_geometry("domain_reference", *domain_reference)
    lanes = run_third_generation_candidates(
        corpora,
        {"general": reference, "domain": domain_geometry},
        rank,
        complete,
        fpr=budget.fpr,
        transfer_threshold=budget.transfer_threshold,
        max_goods_candidates=budget.max_goods_candidates,
        adjudication_budget=adjudication_budget,
        role_samples_per_type=role_samples_per_type,
        seed=budget.seed,
    )
    return _envelope(
        RESEARCH_GENERATION_THIRD,
        _lane_adopted(lanes, "claim_precision"),
        budget.parameters(
            adjudication_budget=adjudication_budget,
            role_samples_per_type=role_samples_per_type,
            adjudicator_model=adjudicator_model,
        ),
        _datasets(corpora, general_reference=reference, domain_reference=domain_geometry),
        rank,
        lanes,
    )


def fourth_generation_summary(
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    complete: "LLMComplete",
    embed: EmbedTexts,
    scorer: "RerankScorer",
    budget: ResearchBudget,
    *,
    adjudicator_model: str,
    cross_encoder_model: str,
    synthesis_per_document: int,
    cross_encoder_rows: int,
) -> JsonObject:
    """In-support control synthesis, cross-encoder relation scoring, conformal tail inference.

    The only generation that builds its control bank from the target corpus itself, so it asks for
    no reference corpus at all.
    """
    lanes = run_fourth_generation_candidates(
        corpora,
        rank,
        complete,
        embed,
        scorer,
        fpr=budget.fpr,
        transfer_threshold=budget.transfer_threshold,
        max_goods_candidates=budget.max_goods_candidates,
        synthesis_per_document=synthesis_per_document,
        cross_encoder_rows=cross_encoder_rows,
        seed=budget.seed,
    )
    return _envelope(
        RESEARCH_GENERATION_FOURTH,
        _lane_adopted(lanes, "conformal"),
        budget.parameters(
            synthesis_per_document=synthesis_per_document,
            cross_encoder_rows=cross_encoder_rows,
            adjudicator_model=adjudicator_model,
            cross_encoder_model=cross_encoder_model,
        ),
        _datasets(corpora),
        rank,
        lanes,
    )
