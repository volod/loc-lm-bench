"""Claim-tier orchestration: optional reranking, calibration, adjudication, and recording."""

import logging
from typing import TYPE_CHECKING

from llb.conflicts.claim.calibration import calibrate_adjudicator, log_calibration
from llb.conflicts.claim.probe import load_calibration_probe
from llb.conflicts.claim.precision import AdjudicatedRow, precision_block
from llb.conflicts.claim.prefilter import (
    ClaimPrefilterRanking,
    prefilter_artifact,
    rank_claim_candidates,
)
from llb.conflicts.constants import TIER_CLAIM, tiers_up_to
from llb.conflicts.models import AuditResult, Finding
from llb.conflicts.store_access import StoreView
from llb.conflicts.tiers.claim import adjudicate_pairs
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.prep.frontier.telemetry import LLMComplete
from llb.rag.rerank import RerankScorer

if TYPE_CHECKING:
    from llb.conflicts.audit import AuditParams

_LOG = logging.getLogger(__name__)


def record_claims(
    result: AuditResult,
    params: "AuditParams",
    store: StoreView,
    governance: dict[str, JsonObject],
    complete: LLMComplete | None,
    claim_scorer: RerankScorer | None,
    semantic_findings: list[Finding],
    pairs: list[tuple[int, int, float]],
) -> None:
    """Adjudicate the selected order and retain every unadjudicated semantic row."""
    if TIER_CLAIM not in tiers_up_to(params.effort):
        result.findings.extend(semantic_findings)
        return
    if complete is None:
        raise SystemExit(
            "[conflicts] the claim tier needs a model endpoint: pass --conflict-model "
            "(and --conflict-backend) so candidate pairs can be adjudicated."
        )
    ranking = _claim_ranking(params, pairs, store.chunks, claim_scorer)
    selected_indexes, cross_encoder_cap = _selected_indexes(params, ranking, len(pairs))
    calibration = _calibrate(params, complete)
    selected = [pairs[index] for index in selected_indexes]
    if len(selected) < len(pairs):
        _LOG.warning(
            "[conflicts] adjudicating %d of %d candidate pairs (--max-claim-pairs)",
            len(selected),
            len(pairs),
        )
    claim_findings, claim_stats, rows = adjudicate_pairs(
        selected, store.chunks, governance, complete
    )
    claim_stats.extra["candidate_rows"] = len(pairs)
    claim_stats.extra["unadjudicated_rows"] = len(pairs) - len(selected)
    result.findings.extend(claim_findings)
    selected_set = set(selected_indexes)
    result.findings.extend(
        finding for index, finding in enumerate(semantic_findings) if index not in selected_set
    )
    result.tiers.append(claim_stats)
    result.claim_precision = precision_block(rows, calibration, seed=params.null_seed)
    result.claim_precision["candidate_order"] = "cross_encoder" if cross_encoder_cap else "cosine"
    _record_prefilter(
        result,
        params,
        ranking,
        selected_indexes,
        rows,
        store.chunks,
        claim_stats.extra,
        cross_encoder_cap,
    )
    _LOG.info("[conflicts] tier=claim findings=%d", len(claim_findings))
    if not result.claim_precision["reported"]:
        _LOG.warning(
            "[conflicts] claim-tier precision not reported: %s", result.claim_precision["reason"]
        )


def _selected_indexes(
    params: "AuditParams",
    ranking: ClaimPrefilterRanking | None,
    candidate_count: int,
) -> tuple[list[int], bool]:
    """Use cross rank only when the explicit cap would actually reduce the candidate list."""
    cross_encoder_cap = bool(
        ranking is not None
        and not ranking.flat_scores
        and 0 < params.max_claim_pairs < candidate_count
    )
    if cross_encoder_cap and ranking is not None:
        order = [candidate.original_index for candidate in ranking.candidates]
    else:
        order = list(range(candidate_count))
    cap = params.max_claim_pairs or candidate_count
    return order[:cap], cross_encoder_cap


def _record_prefilter(
    result: AuditResult,
    params: "AuditParams",
    ranking: ClaimPrefilterRanking | None,
    selected_indexes: list[int],
    rows: list[AdjudicatedRow],
    chunks: list[ChunkRecord],
    tier_extra: JsonObject,
    cross_encoder_cap: bool,
) -> None:
    """Align verdicts to cross rank and attach the complete ranking/cost artifact."""
    if ranking is None:
        return
    rows_by_original_index = dict(zip(selected_indexes, rows, strict=True))
    ranked_rows = [
        rows_by_original_index[candidate.original_index]
        for candidate in ranking.candidates
        if candidate.original_index in rows_by_original_index
    ]
    adjudication_order = "cross_encoder" if cross_encoder_cap else "cosine_evaluation"
    result.claim_prefilter = prefilter_artifact(
        ranking,
        ranked_rows,
        chunks,
        model=params.claim_prefilter_model,
        device=params.claim_prefilter_device,
        adjudication_order=adjudication_order,
    )
    tier_extra.update(
        {
            "claim_prefilter": True,
            "prefilter_scored_rows": len(ranking.candidates),
            "prefilter_seconds": round(ranking.seconds, 3),
            "prefilter_flat_fallback": ranking.flat_scores,
            "prefilter_adjudication_order": adjudication_order,
        }
    )


def _claim_ranking(
    params: "AuditParams",
    pairs: list[tuple[int, int, float]],
    chunks: list[ChunkRecord],
    scorer: RerankScorer | None,
) -> ClaimPrefilterRanking | None:
    """Resolve the optional injected ordering, refusing an enabled stage with no scorer."""
    if not params.claim_prefilter:
        return None
    if scorer is None:
        raise SystemExit(
            "[conflicts] --claim-prefilter needs a cross-encoder scorer; the CLI supplies the "
            "pinned scorer and API callers must inject one"
        )
    ranking = rank_claim_candidates(pairs, chunks, scorer)
    if ranking.flat_scores:
        _LOG.info(
            "[conflicts] claim prefilter scores are flat; preserving the cosine candidate order"
        )
    else:
        _LOG.info(
            "[conflicts] claim prefilter reranked %d candidates in %.3f s",
            len(ranking.candidates),
            ranking.seconds,
        )
    return ranking


def _calibrate(params: "AuditParams", complete: LLMComplete) -> JsonObject | None:
    """Adjudicate the frozen probe first, so the precision block knows what it may print."""
    if not params.calibrate_adjudicator:
        return None
    probe = load_calibration_probe(params.calibration_probe, params.probe_tiers)
    calibration = calibrate_adjudicator(probe, complete)
    log_calibration(calibration)
    return calibration
