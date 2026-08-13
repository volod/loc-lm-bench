"""Cluster-conservative empirical-FDR candidate for conflict-null research."""

import math

from llb.conflicts.null_research_advanced import candidate_gates, clustered_tail_payload
from llb.conflicts.null_research_controls import MatchedControls
from llb.conflicts.interval_stats import wilson_interval
from llb.conflicts.null_research_evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    count_at_or_above,
    fixture_metrics,
    transfer_payload,
)
from llb.conflicts.null_research_geometry import CorpusGeometry
from llb.core.contracts.common import JsonObject


def _cluster_fdr_threshold(
    observed_scores: list[float],
    cluster_maxima: list[float],
    target_fdr: float,
) -> tuple[float, JsonObject]:
    """Maximize selections under a Wilson upper bound over source-cluster exceedances."""
    if not observed_scores or not cluster_maxima:
        raise ValueError("cluster FDR needs observed scores and control clusters")
    candidates = {
        observed_scores[0],
        *(math.nextafter(score, math.inf) for score in cluster_maxima),
    }
    best: tuple[int, float, float, int, float] | None = None
    for threshold in sorted(candidates):
        selected = count_at_or_above(observed_scores, threshold)
        if not selected:
            continue
        exceedances = count_at_or_above(cluster_maxima, threshold)
        _, rate_upper = wilson_interval(exceedances, len(cluster_maxima))
        expected_false_upper = rate_upper * len(observed_scores)
        fdr_upper = min(1.0, expected_false_upper / selected)
        if fdr_upper <= target_fdr:
            proposal = (selected, threshold, fdr_upper, exceedances, expected_false_upper)
            if best is None or proposal[0] > best[0]:
                best = proposal
    if best is None:
        threshold = math.nextafter(max(observed_scores[-1], cluster_maxima[-1]), math.inf)
        _, rate_upper = wilson_interval(0, len(cluster_maxima))
        return threshold, {
            "identified": False,
            "target_fdr": target_fdr,
            "selected_chunk_pairs": 0,
            "cluster_exceedances": 0,
            "cluster_rate_upper_95": round(rate_upper, 9),
            "expected_false_upper": round(rate_upper * len(observed_scores), 3),
            "fdr_upper_95": 1.0,
        }
    selected, threshold, fdr_upper, exceedances, expected_false_upper = best
    _, rate_upper = wilson_interval(exceedances, len(cluster_maxima))
    return threshold, {
        "identified": True,
        "target_fdr": target_fdr,
        "selected_chunk_pairs": selected,
        "cluster_exceedances": exceedances,
        "cluster_rate_upper_95": round(rate_upper, 9),
        "expected_false_upper": round(expected_false_upper, 3),
        "fdr_upper_95": round(fdr_upper, 9),
    }


def build_cluster_fdr_candidate(
    controls: dict[str, MatchedControls],
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    *,
    target_fdr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> JsonObject:
    """Fit and evaluate a matched-control FDR threshold separately on each transfer corpus."""
    fitted = {
        dataset: _cluster_fdr_threshold(
            corpora[dataset].observed_similarities,
            control.cluster_maxima(),
            target_fdr,
        )
        for dataset, control in controls.items()
    }
    thresholds = {dataset: result[0] for dataset, result in fitted.items()}
    diagnostics = {dataset: control.diagnostics for dataset, control in controls.items()}
    tails = {
        dataset: clustered_tail_payload(
            control.scores,
            control.cluster_maxima(),
            thresholds[dataset],
            target_fdr,
            control.effective_units,
            seed + position,
        )
        for position, (dataset, control) in enumerate(controls.items())
    }
    fixture = fixture_metrics(
        corpora["fixture"].document_maxima,
        thresholds["fixture"],
        FIXTURE_POSITIVE_DOC_PAIRS,
    )
    hr = transfer_payload(corpora["hr"].observed_similarities, thresholds["hr"], transfer_threshold)
    goods = transfer_payload(
        corpora["goods"].observed_similarities,
        thresholds["goods"],
        transfer_threshold,
    )
    gates = candidate_gates(
        fixture,
        rank,
        hr,
        goods,
        tails,
        diagnostics,
        max_goods_candidates=max_goods_candidates,
        eligible=True,
        control_key="exchangeable",
    )
    return {
        "method": "surface_matched_cluster_fdr",
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "null_tails": tails,
        "diagnostics": diagnostics,
        "fdr_fits": {dataset: result[1] for dataset, result in fitted.items()},
        "fixture": fixture,
        "hr": hr,
        "goods": goods,
        "gates": gates,
    }
