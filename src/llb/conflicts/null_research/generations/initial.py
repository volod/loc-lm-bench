"""Initial conflict-null candidates retained as the paired research baseline."""

import math

from llb.conflicts.null_research.candidates import (
    build_labelled_candidate,
    build_null_candidate,
)
from llb.conflicts.null_research.evaluation import MIN_TAIL_OBSERVATIONS
from llb.conflicts.null_research.geometry import (
    CorpusGeometry,
    EmbedTexts,
    cross_corpus_null,
    held_out_document_null,
    permutation_null,
)
from llb.core.contracts.common import JsonObject

RESEARCH_METHODS = (
    "cross_corpus",
    "token_permutation",
    "sentence_permutation",
    "held_out_document",
    "labelled_calibration",
)


def _unique_text_units(geometry: CorpusGeometry) -> int:
    return len({geometry.chunks[index]["text"] for index in geometry.allowed})


def _resolved_permutations(
    corpora: dict[str, CorpusGeometry], requested: int, fpr: float
) -> dict[str, int]:
    """Raise repeats until the pair-row tail, but not its source clusters, is populated."""
    return {
        name: max(
            requested,
            math.ceil(MIN_TAIL_OBSERVATIONS / (fpr * len(corpus.allowed))),
        )
        for name, corpus in corpora.items()
    }


def _permutation_candidates(
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    embed: EmbedTexts,
    repeats: dict[str, int],
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> list[JsonObject]:
    methods: list[JsonObject] = []
    for mode in ("token", "sentence"):
        scores = {
            name: permutation_null(
                corpus,
                embed,
                mode=mode,
                permutations=repeats[name],
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
                effective_units={
                    name: _unique_text_units(corpus) for name, corpus in corpora.items()
                },
            )
        )
    return methods


def run_initial_candidates(
    corpora: dict[str, CorpusGeometry],
    reference: CorpusGeometry,
    rank: JsonObject,
    embed: EmbedTexts,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    permutations: int,
    seed: int,
) -> tuple[list[JsonObject], dict[str, int]]:
    """Evaluate the original five candidates with corrected effective-unit tail gates."""
    reference_units = _unique_text_units(reference)
    methods = [
        build_null_candidate(
            "cross_corpus",
            {name: cross_corpus_null(corpus, reference) for name, corpus in corpora.items()},
            corpora,
            rank,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            eligible_as_null=True,
            effective_units={
                name: min(_unique_text_units(corpus), reference_units)
                for name, corpus in corpora.items()
            },
        )
    ]
    repeats = _resolved_permutations(corpora, permutations, fpr)
    methods.extend(
        _permutation_candidates(
            corpora,
            rank,
            embed,
            repeats,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            seed=seed,
        )
    )
    methods.append(
        build_null_candidate(
            "held_out_document",
            {name: held_out_document_null(corpus) for name, corpus in corpora.items()},
            corpora,
            rank,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            eligible_as_null=False,
            effective_units={
                name: len({corpus.chunks[index]["doc_id"] for index in corpus.allowed})
                for name, corpus in corpora.items()
            },
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
    return methods, repeats
