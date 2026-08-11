"""Execute and decide the independent-null research matrix."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from llb.conflicts.constants import (
    RESEARCH_GENERATION_INITIAL,
    RESEARCH_GENERATION_NEXT,
    RESEARCH_GENERATION_THIRD,
    RESEARCH_GENERATIONS,
)
from llb.conflicts.null_research_evaluation import (
    DEFAULT_MAX_GOODS_CANDIDATES,
    DEFAULT_RESEARCH_FPR,
    DEFAULT_RESEARCH_RANK_BUDGET,
    DEFAULT_TRANSFER_THRESHOLD,
    FIXTURE_POSITIVE_DOC_PAIRS,
    rank_baseline,
)
from llb.conflicts.null_research_geometry import (
    CorpusGeometry,
    geometry_payload,
    prepare_geometry,
)
from llb.conflicts.null_research_initial import RESEARCH_METHODS, run_initial_candidates
from llb.conflicts.null_research_nextgen import run_next_generation_candidates
from llb.conflicts.null_research_third import run_third_generation_candidates
from llb.conflicts.store_access import StoreView
from llb.core.contracts.common import JsonObject

if TYPE_CHECKING:
    from llb.prep.frontier_telemetry import LLMComplete

EmbedTexts = Callable[[list[str]], object]

__all__ = ["RESEARCH_METHODS", "run_null_research"]


def _adopted_methods(methods: list[JsonObject]) -> list[str]:
    return [str(method["method"]) for method in methods if method["gates"]["accepted"]]


def _next_generation_summary(
    corpora: dict[str, CorpusGeometry],
    reference: CorpusGeometry,
    domain_reference: tuple[Path, StoreView],
    rank: JsonObject,
    embed: EmbedTexts,
    *,
    fpr: float,
    rank_budget: int,
    transfer_threshold: float,
    max_goods_candidates: int,
    matches_per_reference: int,
    seed: int,
) -> JsonObject:
    domain_geometry = prepare_geometry("domain_reference", *domain_reference)
    methods, traces = run_next_generation_candidates(
        corpora,
        {"general": reference, "domain": domain_geometry},
        rank,
        embed,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        matches_per_reference=matches_per_reference,
        seed=seed,
    )
    adopted = _adopted_methods(methods)
    return {
        "research_generation": RESEARCH_GENERATION_NEXT,
        "verdict": "adopt" if adopted else "negative",
        "adopted_methods": adopted,
        "parameters": {
            "fpr": fpr,
            "rank_budget": rank_budget,
            "transfer_threshold": transfer_threshold,
            "max_goods_candidates": max_goods_candidates,
            "matches_per_reference": matches_per_reference,
            "seed": seed,
        },
        "datasets": {
            **{name: geometry_payload(corpus) for name, corpus in corpora.items()},
            "general_reference": geometry_payload(reference),
            "domain_reference": geometry_payload(domain_geometry),
        },
        "rank_baseline": rank,
        "methods": methods,
        "control_traces": traces,
    }


def _initial_summary(
    corpora: dict[str, CorpusGeometry],
    reference: CorpusGeometry,
    rank: JsonObject,
    embed: EmbedTexts,
    *,
    fpr: float,
    rank_budget: int,
    transfer_threshold: float,
    max_goods_candidates: int,
    permutations: int,
    seed: int,
) -> JsonObject:
    methods, repeats = run_initial_candidates(
        corpora,
        reference,
        rank,
        embed,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        permutations=permutations,
        seed=seed,
    )
    adopted = _adopted_methods(methods)
    return {
        "research_generation": RESEARCH_GENERATION_INITIAL,
        "verdict": "adopt" if adopted else "negative",
        "adopted_methods": adopted,
        "parameters": {
            "fpr": fpr,
            "rank_budget": rank_budget,
            "transfer_threshold": transfer_threshold,
            "max_goods_candidates": max_goods_candidates,
            "minimum_permutations": permutations,
            "resolved_permutations": repeats,
            "seed": seed,
        },
        "datasets": {
            **{name: geometry_payload(corpus) for name, corpus in corpora.items()},
            "reference": geometry_payload(reference),
        },
        "rank_baseline": rank,
        "methods": methods,
    }


def _third_generation_summary(
    corpora: dict[str, CorpusGeometry],
    reference: CorpusGeometry,
    domain_reference: tuple[Path, StoreView],
    rank: JsonObject,
    complete: "LLMComplete",
    *,
    adjudicator_model: str,
    fpr: float,
    rank_budget: int,
    transfer_threshold: float,
    max_goods_candidates: int,
    adjudication_budget: int,
    role_samples_per_type: int,
    seed: int,
) -> JsonObject:
    domain_geometry = prepare_geometry("domain_reference", *domain_reference)
    lanes = run_third_generation_candidates(
        corpora,
        {"general": reference, "domain": domain_geometry},
        rank,
        complete,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        adjudication_budget=adjudication_budget,
        role_samples_per_type=role_samples_per_type,
        seed=seed,
    )
    methods = lanes["methods"]
    precision = lanes["claim_precision"]
    assert isinstance(methods, list) and isinstance(precision, dict)
    adopted = _adopted_methods(methods)
    if precision["gates"]["accepted"]:
        adopted.append(str(precision["method"]))
    return {
        "research_generation": RESEARCH_GENERATION_THIRD,
        "verdict": "adopt" if adopted else "negative",
        "adopted_methods": adopted,
        "parameters": {
            "fpr": fpr,
            "rank_budget": rank_budget,
            "transfer_threshold": transfer_threshold,
            "max_goods_candidates": max_goods_candidates,
            "adjudication_budget": adjudication_budget,
            "role_samples_per_type": role_samples_per_type,
            "adjudicator_model": adjudicator_model,
            "seed": seed,
        },
        "datasets": {
            **{name: geometry_payload(corpus) for name, corpus in corpora.items()},
            "general_reference": geometry_payload(reference),
            "domain_reference": geometry_payload(domain_geometry),
        },
        "rank_baseline": rank,
        **lanes,
    }


def run_null_research(
    *,
    fixture: tuple[Path, StoreView],
    hr: tuple[Path, StoreView],
    goods: tuple[Path, StoreView],
    reference: tuple[Path, StoreView],
    embed: EmbedTexts | None = None,
    domain_reference: tuple[Path, StoreView] | None = None,
    generation: str = RESEARCH_GENERATION_INITIAL,
    complete: "LLMComplete | None" = None,
    fpr: float = DEFAULT_RESEARCH_FPR,
    rank_budget: int = DEFAULT_RESEARCH_RANK_BUDGET,
    transfer_threshold: float = DEFAULT_TRANSFER_THRESHOLD,
    max_goods_candidates: int = DEFAULT_MAX_GOODS_CANDIDATES,
    permutations: int = 3,
    matches_per_reference: int = 2,
    adjudication_budget: int = 50,
    role_samples_per_type: int = 12,
    adjudicator_model: str = "",
    seed: int = 0,
) -> JsonObject:
    """Evaluate all proposed nulls and return a complete adopt-or-reject record."""
    if generation not in RESEARCH_GENERATIONS:
        raise ValueError(
            f"unknown research generation {generation!r}; expected one of {RESEARCH_GENERATIONS}"
        )
    corpora = {
        name: prepare_geometry(name, corpus, store)
        for name, (corpus, store) in {"fixture": fixture, "hr": hr, "goods": goods}.items()
    }
    reference_geometry = prepare_geometry("reference", *reference)
    rank = rank_baseline(
        corpora["fixture"].observed_similarities,
        corpora["fixture"].document_maxima,
        FIXTURE_POSITIVE_DOC_PAIRS,
        rank_budget,
    )
    if generation != RESEARCH_GENERATION_INITIAL and domain_reference is None:
        raise ValueError(f"{generation}-generation research requires a domain reference corpus")
    if generation != RESEARCH_GENERATION_THIRD and embed is None:
        raise ValueError(f"{generation}-generation research requires an embedder")
    if generation == RESEARCH_GENERATION_THIRD:
        if complete is None:
            raise ValueError("third-generation research requires a claim adjudication model")
        assert domain_reference is not None
        return _third_generation_summary(
            corpora,
            reference_geometry,
            domain_reference,
            rank,
            complete,
            adjudicator_model=adjudicator_model,
            fpr=fpr,
            rank_budget=rank_budget,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            adjudication_budget=adjudication_budget,
            role_samples_per_type=role_samples_per_type,
            seed=seed,
        )
    if generation == RESEARCH_GENERATION_NEXT:
        assert domain_reference is not None and embed is not None
        return _next_generation_summary(
            corpora,
            reference_geometry,
            domain_reference,
            rank,
            embed,
            fpr=fpr,
            rank_budget=rank_budget,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            matches_per_reference=matches_per_reference,
            seed=seed,
        )
    assert embed is not None
    return _initial_summary(
        corpora,
        reference_geometry,
        rank,
        embed,
        fpr=fpr,
        rank_budget=rank_budget,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        permutations=permutations,
        seed=seed,
    )
