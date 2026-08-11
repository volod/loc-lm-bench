"""Execute and decide the independent-null research matrix."""

from collections.abc import Callable
import math
from pathlib import Path

from llb.conflicts.null_research_candidates import (
    build_labelled_candidate,
    build_null_candidate,
)
from llb.conflicts.null_research_evaluation import (
    DEFAULT_MAX_GOODS_CANDIDATES,
    DEFAULT_RESEARCH_FPR,
    DEFAULT_RESEARCH_RANK_BUDGET,
    DEFAULT_TRANSFER_THRESHOLD,
    FIXTURE_POSITIVE_DOC_PAIRS,
    MIN_TAIL_OBSERVATIONS,
    rank_baseline,
)
from llb.conflicts.null_research_geometry import (
    CorpusGeometry,
    cross_corpus_null,
    held_out_document_null,
    permutation_null,
    prepare_geometry,
)
from llb.conflicts.store_access import StoreView
from llb.core.contracts.common import JsonObject

EmbedTexts = Callable[[list[str]], object]

RESEARCH_METHODS = (
    "cross_corpus",
    "token_permutation",
    "sentence_permutation",
    "held_out_document",
    "labelled_calibration",
)


def _geometry_payload(geometry: CorpusGeometry) -> JsonObject:
    return {
        "corpus_root": geometry.corpus_root,
        "store_dir": str(geometry.store_dir),
        "embedding_model": geometry.embedding_model,
        "corpus_fingerprint": geometry.corpus_fingerprint,
        "dimensions": geometry.vectors.dim,
        "chunks": len(geometry.chunks),
        "comparable_chunks": len(geometry.allowed),
        "documents": len({geometry.chunks[index]["doc_id"] for index in geometry.allowed}),
        "comparable_chunk_pairs": len(geometry.observed_similarities),
        "document_pairs": len(geometry.document_maxima),
        "centered": geometry.centered,
        **geometry.excluded,
    }


def _resolved_permutations(
    corpora: dict[str, CorpusGeometry], requested: int, fpr: float
) -> dict[str, int]:
    """Raise small-corpus repeats until the nominal tail has enough observations."""
    return {
        name: max(
            requested,
            math.ceil(MIN_TAIL_OBSERVATIONS / (fpr * len(corpus.allowed))),
        )
        for name, corpus in corpora.items()
    }


def run_null_research(
    *,
    fixture: tuple[Path, StoreView],
    hr: tuple[Path, StoreView],
    goods: tuple[Path, StoreView],
    reference: tuple[Path, StoreView],
    embed: EmbedTexts,
    fpr: float = DEFAULT_RESEARCH_FPR,
    rank_budget: int = DEFAULT_RESEARCH_RANK_BUDGET,
    transfer_threshold: float = DEFAULT_TRANSFER_THRESHOLD,
    max_goods_candidates: int = DEFAULT_MAX_GOODS_CANDIDATES,
    permutations: int = 3,
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
    cross_scores = {
        name: cross_corpus_null(corpus, reference_geometry) for name, corpus in corpora.items()
    }
    methods = [
        build_null_candidate(
            "cross_corpus",
            cross_scores,
            corpora,
            rank,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            eligible_as_null=True,
        )
    ]
    permutation_repeats = _resolved_permutations(corpora, permutations, fpr)
    for mode in ("token", "sentence"):
        scores = {
            name: permutation_null(
                corpus,
                embed,
                mode=mode,
                permutations=permutation_repeats[name],
                seed=seed,
            )
            for name, corpus in corpora.items()
        }
        methods.append(
            build_null_candidate(
                f"{mode}_permutation",
                scores,
                corpora,
                rank,
                fpr=fpr,
                transfer_threshold=transfer_threshold,
                max_goods_candidates=max_goods_candidates,
                eligible_as_null=True,
            )
        )
    held_out = {name: held_out_document_null(corpus) for name, corpus in corpora.items()}
    methods.append(
        build_null_candidate(
            "held_out_document",
            held_out,
            corpora,
            rank,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            eligible_as_null=False,
        )
    )
    methods.append(
        build_labelled_candidate(
            corpora,
            rank,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
        )
    )
    adopted = [str(method["method"]) for method in methods if method["gates"]["accepted"]]
    return {
        "verdict": "adopt" if adopted else "negative",
        "adopted_methods": adopted,
        "parameters": {
            "fpr": fpr,
            "rank_budget": rank_budget,
            "transfer_threshold": transfer_threshold,
            "max_goods_candidates": max_goods_candidates,
            "minimum_permutations": permutations,
            "resolved_permutations": permutation_repeats,
            "seed": seed,
        },
        "datasets": {
            **{name: _geometry_payload(corpus) for name, corpus in corpora.items()},
            "reference": _geometry_payload(reference_geometry),
        },
        "rank_baseline": rank,
        "methods": methods,
    }
