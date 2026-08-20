"""Candidate result assembly and acceptance gates for conflict-null research."""

from llb.conflicts.null_research.evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    MIN_TAIL_OBSERVATIONS,
    fit_labelled_threshold,
    fixture_metrics,
    null_tail_payload,
    threshold_for_fpr,
    transfer_payload,
)
from llb.conflicts.null_research.geometry import CorpusGeometry
from llb.core.contracts.common import JsonObject


def _gate_payload(
    fixture: JsonObject,
    rank: JsonObject,
    hr: JsonObject,
    goods: JsonObject,
    *,
    max_goods_candidates: int,
    tail_resolved: bool,
    eligible_as_null: bool,
) -> JsonObject:
    gates = {
        "beats_rank_fixture_f1": float(fixture["f1"]) > float(rank["f1"]),
        "recovers_hr_baseline": float(hr["baseline_recall"]) >= 1.0,
        "does_not_flood_goods": int(goods["selected_chunk_pairs"]) <= max_goods_candidates,
        "tail_resolved": tail_resolved,
        "eligible_as_independent_null": eligible_as_null,
    }
    return {**gates, "accepted": all(gates.values())}


def build_null_candidate(
    name: str,
    scores: dict[str, list[float]],
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    eligible_as_null: bool,
    effective_units: dict[str, int],
) -> JsonObject:
    thresholds = {dataset: threshold_for_fpr(values, fpr) for dataset, values in scores.items()}
    tails = {
        dataset: null_tail_payload(scores[dataset], thresholds[dataset], fpr) for dataset in scores
    }
    for dataset, tail in tails.items():
        pair_row_resolved = bool(tail["tail_resolved"])
        expected_independent = fpr * effective_units[dataset]
        tail.update(
            {
                "pair_row_tail_resolved": pair_row_resolved,
                "effective_independent_units": effective_units[dataset],
                "expected_independent_tail_observations": round(expected_independent, 3),
                "tail_resolved": expected_independent >= MIN_TAIL_OBSERVATIONS,
            }
        )
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
    gates = _gate_payload(
        fixture,
        rank,
        hr,
        goods,
        max_goods_candidates=max_goods_candidates,
        tail_resolved=all(bool(tail["tail_resolved"]) for tail in tails.values()),
        eligible_as_null=eligible_as_null,
    )
    return {
        "method": name,
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "null_tails": tails,
        "fixture": fixture,
        "hr": hr,
        "goods": goods,
        "gates": gates,
    }


def build_labelled_candidate(
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    *,
    transfer_threshold: float,
    max_goods_candidates: int,
) -> JsonObject:
    threshold, fixture = fit_labelled_threshold(
        corpora["fixture"].document_maxima, FIXTURE_POSITIVE_DOC_PAIRS
    )
    hr = transfer_payload(corpora["hr"].observed_similarities, threshold, transfer_threshold)
    goods = transfer_payload(corpora["goods"].observed_similarities, threshold, transfer_threshold)
    gates = _gate_payload(
        fixture,
        rank,
        hr,
        goods,
        max_goods_candidates=max_goods_candidates,
        tail_resolved=False,
        eligible_as_null=False,
    )
    return {
        "method": "labelled_calibration",
        "thresholds": {dataset: round(threshold, 6) for dataset in corpora},
        "null_tails": {},
        "fixture": fixture,
        "hr": hr,
        "goods": goods,
        "gates": gates,
        "limitation": "supervised fixture fit; it does not estimate an independent FPR",
    }
