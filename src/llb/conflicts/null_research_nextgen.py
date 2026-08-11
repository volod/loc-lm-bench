"""Orchestration for the second-generation conflict-null research matrix."""

from llb.conflicts.null_research_advanced import (
    ADVANCED_RESEARCH_METHODS,
    _fpr_candidate,
    _residual_space,
)
from llb.conflicts.null_research_controls import MatchedControls, build_matched_controls
from llb.conflicts.null_research_counterfactuals import (
    CounterfactualControls,
    build_counterfactual_controls,
)
from llb.conflicts.null_research_fdr import build_cluster_fdr_candidate
from llb.conflicts.null_research_geometry import CorpusGeometry, EmbedTexts
from llb.core.contracts.common import JsonObject

__all__ = ["ADVANCED_RESEARCH_METHODS", "run_next_generation_candidates"]


def _matched_candidates(
    matched: dict[str, MatchedControls],
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> list[JsonObject]:
    diagnostics = {dataset: control.diagnostics for dataset, control in matched.items()}
    effective_units = {dataset: control.effective_units for dataset, control in matched.items()}
    raw = _fpr_candidate(
        "surface_matched_reference",
        {dataset: control.scores for dataset, control in matched.items()},
        {dataset: control.cluster_maxima() for dataset, control in matched.items()},
        effective_units,
        diagnostics,
        corpora,
        rank,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        seed=seed,
        eligible=True,
        control_key="exchangeable",
    )
    residual_spaces = {
        dataset: _residual_space(corpus, matched[dataset]) for dataset, corpus in corpora.items()
    }
    residual = _fpr_candidate(
        "surface_matched_residual",
        {dataset: control.residual_scores() for dataset, control in matched.items()},
        {dataset: control.cluster_maxima(residual=True) for dataset, control in matched.items()},
        effective_units,
        diagnostics,
        corpora,
        rank,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        seed=seed + len(corpora),
        eligible=True,
        control_key="exchangeable",
        residual_spaces=residual_spaces,
    )
    fdr = build_cluster_fdr_candidate(
        matched,
        corpora,
        rank,
        target_fdr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        seed=seed + 2 * len(corpora),
    )
    return [raw, residual, fdr]


def _counterfactual_candidate(
    counterfactuals: dict[str, CounterfactualControls],
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> JsonObject:
    return _fpr_candidate(
        "claim_counterfactual_control",
        {dataset: control.scores for dataset, control in counterfactuals.items()},
        {dataset: control.cluster_maxima() for dataset, control in counterfactuals.items()},
        {dataset: control.effective_units for dataset, control in counterfactuals.items()},
        {dataset: control.diagnostics for dataset, control in counterfactuals.items()},
        corpora,
        rank,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        seed=seed + 3 * len(corpora),
        eligible=False,
        control_key="independent_semantic_verification",
    )


def run_next_generation_candidates(
    corpora: dict[str, CorpusGeometry],
    references: dict[str, CorpusGeometry],
    rank: JsonObject,
    embed: EmbedTexts,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    matches_per_reference: int,
    seed: int,
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Build shared controls once, then evaluate four distinct threshold hypotheses."""
    matched = {
        dataset: build_matched_controls(
            corpus,
            references,
            matches_per_reference=matches_per_reference,
        )
        for dataset, corpus in corpora.items()
    }
    methods = _matched_candidates(
        matched,
        corpora,
        rank,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        seed=seed,
    )
    counterfactuals = {
        dataset: build_counterfactual_controls(corpus, embed) for dataset, corpus in corpora.items()
    }
    methods.append(
        _counterfactual_candidate(
            counterfactuals,
            corpora,
            rank,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            seed=seed,
        )
    )
    traces = [trace for dataset in corpora for trace in counterfactuals[dataset].traces]
    assert [str(method["method"]) for method in methods] == list(ADVANCED_RESEARCH_METHODS)
    return methods, traces
