"""Second-generation matched, residual, FDR, and counterfactual null candidates."""

from collections import defaultdict
from dataclasses import dataclass

from llb.conflicts.null_research_controls import MatchedControls
from llb.conflicts.null_research_evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    MIN_COVERAGE_PROBABILITY,
    MIN_TAIL_OBSERVATIONS,
    count_at_or_above,
    fixture_metrics,
    null_tail_payload,
    simulated_wilson_coverage,
    threshold_for_fpr,
    transfer_payload,
    wilson_interval,
)
from llb.conflicts.null_research_geometry import CorpusGeometry, DocPair
from llb.core.contracts.common import JsonObject

ADVANCED_RESEARCH_METHODS = (
    "surface_matched_reference",
    "surface_matched_residual",
    "surface_matched_cluster_fdr",
    "claim_counterfactual_control",
)


@dataclass(frozen=True)
class _ResidualSpace:
    sorted_scores: list[float]
    raw_scores: list[float]
    scores: list[float]
    document_maxima: dict[DocPair, float]


def _residual_pair_scores(
    corpus: CorpusGeometry, controls: MatchedControls
) -> tuple[list[float], list[float]]:
    groups = [corpus.chunks[index]["doc_id"] for index in corpus.allowed]
    doc_codes = {doc_id: code for code, doc_id in enumerate(sorted(set(groups)))}
    raw_scores = corpus.vectors.cross_group_similarities(
        corpus.allowed,
        [doc_codes[group] for group in groups],
    )
    offsets = [
        (controls.source_centers[left] + controls.source_centers[right]) / 2.0
        for position, left in enumerate(corpus.allowed)
        for right in corpus.allowed[position + 1 :]
        if corpus.chunks[left]["doc_id"] != corpus.chunks[right]["doc_id"]
    ]
    if len(raw_scores) != len(offsets):
        raise AssertionError("residual pair ordering differs from the vector scan")
    return raw_scores, [score - offset for score, offset in zip(raw_scores, offsets)]


def _residual_document_maxima(
    corpus: CorpusGeometry, controls: MatchedControls
) -> dict[DocPair, float]:
    by_doc: dict[str, list[int]] = defaultdict(list)
    for ordinal in corpus.allowed:
        by_doc[corpus.chunks[ordinal]["doc_id"]].append(ordinal)
    maxima: dict[DocPair, float] = {}
    doc_ids = sorted(by_doc)
    for position, left_doc in enumerate(doc_ids):
        for right_doc in doc_ids[position + 1 :]:
            values = corpus.vectors.cross_similarities(
                corpus.vectors, by_doc[left_doc], by_doc[right_doc]
            )
            # The explicit product keeps offsets aligned with cross_similarities' left-major order.
            residuals = [
                value - (controls.source_centers[left] + controls.source_centers[right]) / 2.0
                for value, (left, right) in zip(
                    values,
                    ((left, right) for left in by_doc[left_doc] for right in by_doc[right_doc]),
                )
            ]
            maxima[(left_doc, right_doc)] = max(residuals)
    return maxima


def _residual_space(corpus: CorpusGeometry, controls: MatchedControls) -> _ResidualSpace:
    raw_scores, scores = _residual_pair_scores(corpus, controls)
    return _ResidualSpace(
        sorted(scores),
        raw_scores,
        scores,
        _residual_document_maxima(corpus, controls),
    )


def _residual_transfer(
    space: _ResidualSpace, threshold: float, baseline_threshold: float
) -> JsonObject:
    selected = count_at_or_above(space.sorted_scores, threshold)
    baseline = sum(score >= baseline_threshold for score in space.raw_scores)
    recovered = sum(
        raw >= baseline_threshold and residual >= threshold
        for raw, residual in zip(space.raw_scores, space.scores)
    )
    return {
        "threshold": round(threshold, 6),
        "selected_chunk_pairs": selected,
        "baseline_threshold": baseline_threshold,
        "baseline_chunk_pairs": baseline,
        "baseline_recovered": recovered,
        "baseline_recall": round(recovered / baseline, 6) if baseline else 1.0,
    }


def _tail_payload(
    scores: list[float],
    cluster_maxima: list[float],
    threshold: float,
    fpr: float,
    effective_units: int,
    seed: int,
) -> JsonObject:
    payload = null_tail_payload(scores, threshold, fpr)
    pair_tail_resolved = bool(payload["tail_resolved"])
    cluster_exceedances = count_at_or_above(cluster_maxima, threshold)
    cluster_lower, cluster_upper = wilson_interval(cluster_exceedances, len(cluster_maxima))
    coverage = simulated_wilson_coverage(effective_units, fpr, seed=seed)
    expected_independent = fpr * effective_units
    payload.update(
        {
            "pair_row_tail_resolved": pair_tail_resolved,
            "source_clusters": len(cluster_maxima),
            "effective_independent_units": effective_units,
            "cluster_exceedances": cluster_exceedances,
            "cluster_tail_wilson_95": [round(cluster_lower, 9), round(cluster_upper, 9)],
            "expected_independent_tail_observations": round(expected_independent, 3),
            "wilson_coverage_simulations": round(coverage, 6),
            "coverage_valid": coverage >= MIN_COVERAGE_PROBABILITY,
            "tail_resolved": expected_independent >= MIN_TAIL_OBSERVATIONS,
        }
    )
    return payload


def _gates(
    fixture: JsonObject,
    rank: JsonObject,
    hr: JsonObject,
    goods: JsonObject,
    tails: dict[str, JsonObject],
    diagnostics: dict[str, JsonObject],
    *,
    max_goods_candidates: int,
    eligible: bool,
    control_key: str,
) -> JsonObject:
    gates = {
        "beats_rank_fixture_f1": float(fixture["f1"]) > float(rank["f1"]),
        "recovers_hr_baseline": float(hr["baseline_recall"]) >= 1.0,
        "does_not_flood_goods": int(goods["selected_chunk_pairs"]) <= max_goods_candidates,
        "tail_resolved": all(bool(tail["tail_resolved"]) for tail in tails.values()),
        "simulation_coverage": all(bool(tail["coverage_valid"]) for tail in tails.values()),
        "controls_valid": all(bool(payload[control_key]) for payload in diagnostics.values()),
        "eligible_as_independent_null": eligible,
    }
    return {**gates, "accepted": all(gates.values())}


def _fpr_candidate(
    name: str,
    control_scores: dict[str, list[float]],
    cluster_maxima: dict[str, list[float]],
    effective_units: dict[str, int],
    diagnostics: dict[str, JsonObject],
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
    eligible: bool,
    control_key: str,
    residual_spaces: dict[str, _ResidualSpace] | None = None,
) -> JsonObject:
    thresholds = {
        dataset: threshold_for_fpr(scores, fpr) for dataset, scores in control_scores.items()
    }
    tails = {
        dataset: _tail_payload(
            control_scores[dataset],
            cluster_maxima[dataset],
            thresholds[dataset],
            fpr,
            effective_units[dataset],
            seed + position,
        )
        for position, dataset in enumerate(control_scores)
    }
    if residual_spaces is None:
        fixture = fixture_metrics(
            corpora["fixture"].document_maxima,
            thresholds["fixture"],
            FIXTURE_POSITIVE_DOC_PAIRS,
        )
        hr = transfer_payload(
            corpora["hr"].observed_similarities, thresholds["hr"], transfer_threshold
        )
        goods = transfer_payload(
            corpora["goods"].observed_similarities, thresholds["goods"], transfer_threshold
        )
    else:
        fixture = fixture_metrics(
            residual_spaces["fixture"].document_maxima,
            thresholds["fixture"],
            FIXTURE_POSITIVE_DOC_PAIRS,
        )
        hr = _residual_transfer(residual_spaces["hr"], thresholds["hr"], transfer_threshold)
        goods = _residual_transfer(
            residual_spaces["goods"], thresholds["goods"], transfer_threshold
        )
    gates = _gates(
        fixture,
        rank,
        hr,
        goods,
        tails,
        diagnostics,
        max_goods_candidates=max_goods_candidates,
        eligible=eligible,
        control_key=control_key,
    )
    return {
        "method": name,
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "null_tails": tails,
        "diagnostics": diagnostics,
        "fixture": fixture,
        "hr": hr,
        "goods": goods,
        "gates": gates,
    }
