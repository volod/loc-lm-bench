"""Execute and decide the independent-null research matrix."""

from collections.abc import Callable
from pathlib import Path

from llb.conflicts.null_research_evaluation import (
    DEFAULT_MAX_GOODS_CANDIDATES,
    DEFAULT_RESEARCH_FPR,
    DEFAULT_RESEARCH_RANK_BUDGET,
    DEFAULT_TRANSFER_THRESHOLD,
    FIXTURE_POSITIVE_DOC_PAIRS,
    rank_baseline,
)
from llb.conflicts.null_research_geometry import CorpusGeometry, prepare_geometry
from llb.conflicts.null_research_initial import (
    RESEARCH_METHODS as RESEARCH_METHODS,
    run_initial_candidates,
)
from llb.conflicts.store_access import StoreView
from llb.core.contracts.common import JsonObject

EmbedTexts = Callable[[list[str]], object]


def _geometry_payload(geometry: CorpusGeometry) -> JsonObject:
    return {
        "corpus_root": geometry.corpus_root,
        "store_dir": str(geometry.store_dir),
        "embedding_model": geometry.embedding_model,
        "corpus_fingerprint": geometry.corpus_fingerprint,
        "dimensions": geometry.vectors.dim,
        "chunks": len(geometry.chunks),
        "comparable_chunks": len(geometry.allowed),
        "unique_comparable_texts": len(
            {geometry.chunks[index]["text"] for index in geometry.allowed}
        ),
        "documents": len({geometry.chunks[index]["doc_id"] for index in geometry.allowed}),
        "comparable_chunk_pairs": len(geometry.observed_similarities),
        "document_pairs": len(geometry.document_maxima),
        "centered": geometry.centered,
        **geometry.excluded,
    }


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
    from llb.conflicts.null_research_nextgen import run_next_generation_candidates

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
        "research_generation": "next",
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
            **{name: _geometry_payload(corpus) for name, corpus in corpora.items()},
            "general_reference": _geometry_payload(reference),
            "domain_reference": _geometry_payload(domain_geometry),
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
        "research_generation": "initial",
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
            **{name: _geometry_payload(corpus) for name, corpus in corpora.items()},
            "reference": _geometry_payload(reference),
        },
        "rank_baseline": rank,
        "methods": methods,
    }


def run_null_research(
    *,
    fixture: tuple[Path, StoreView],
    hr: tuple[Path, StoreView],
    goods: tuple[Path, StoreView],
    reference: tuple[Path, StoreView],
    embed: EmbedTexts,
    domain_reference: tuple[Path, StoreView] | None = None,
    next_generation: bool = False,
    fpr: float = DEFAULT_RESEARCH_FPR,
    rank_budget: int = DEFAULT_RESEARCH_RANK_BUDGET,
    transfer_threshold: float = DEFAULT_TRANSFER_THRESHOLD,
    max_goods_candidates: int = DEFAULT_MAX_GOODS_CANDIDATES,
    permutations: int = 3,
    matches_per_reference: int = 2,
    seed: int = 0,
) -> JsonObject:
    """Evaluate all proposed nulls and return a complete adopt-or-reject record."""
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
    if next_generation:
        if domain_reference is None:
            raise ValueError("next-generation research requires a domain reference corpus")
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
