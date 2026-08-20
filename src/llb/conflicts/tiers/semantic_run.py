"""Semantic and claim-tier orchestration, split from the cheaper audit tiers."""

import logging
from typing import TYPE_CHECKING

from llb.conflicts.bundle.record import RunInputs
from llb.conflicts.bundle.candidate_record import DEFAULT_CANDIDATE_RECORD_PAIRS, CandidateRecord
from llb.conflicts.claim.calibration import calibrate_adjudicator, load_calibration_probe
from llb.conflicts.claim.precision import precision_block
from llb.conflicts.tiers.claim import adjudicate_pairs
from llb.conflicts.constants import (
    DEFAULT_COSINE_THRESHOLD,
    MIN_CENTERING_VECTORS,
    TIER_CLAIM,
    tiers_up_to,
)
from llb.conflicts.corpus import CorpusDoc
from llb.conflicts.bundle.document_chunks import DocumentChunks
from llb.conflicts.bundle.document_exclusions import DocumentExclusions
from llb.conflicts.models import AuditResult, Finding
from llb.conflicts.needles import analyze_needles
from llb.conflicts.calibration.operating_point import resolve_cos_threshold
from llb.conflicts.calibration.sampling import estimate_null_distribution
from llb.conflicts.semantic_tree.projected_index import prepare_projected_index
from llb.conflicts.tiers.semantic_filter import select_content_chunks
from llb.conflicts.tiers.semantic import build_tree, detect_semantic_pairs
from llb.conflicts.store_access import StoreView
from llb.conflicts.semantic_tree.tree import SemanticPrefixTree
from llb.conflicts.semantic_tree.refresh import tree_meta
from llb.conflicts.semantic_tree.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.goldset.schema import GoldItem
from llb.prep.frontier.telemetry import LLMComplete

if TYPE_CHECKING:
    from llb.conflicts.audit import AuditParams

_LOG = logging.getLogger(__name__)


def _calibrate_threshold(
    params: "AuditParams",
    vectors: VectorSet,
    chunks: list[ChunkRecord],
    allowed: set[int],
) -> tuple[float, JsonObject]:
    """Resolve the operating cosine and its optional null-distribution record."""
    distribution = None
    if params.cos_quantile is not None or params.max_candidate_pairs is not None:
        distribution = estimate_null_distribution(
            vectors,
            chunks,
            allowed,
            sample_pairs=params.null_sample_pairs,
            seed=params.null_seed,
        )
    threshold, source, resolved_quantile = resolve_cos_threshold(
        explicit=params.cos_threshold,
        quantile=params.cos_quantile,
        default=DEFAULT_COSINE_THRESHOLD,
        distribution=distribution,
        max_candidate_pairs=params.max_candidate_pairs,
    )
    payload: JsonObject = {
        "cos_threshold": threshold,
        "cos_threshold_source": source,
    }
    if distribution is not None:
        payload["null_distribution"] = distribution.payload(
            resolved_quantile, params.max_candidate_pairs
        )
    return threshold, payload


def _active_tree(
    params: "AuditParams",
    store: StoreView,
    vectors: VectorSet,
    tree: SemanticPrefixTree | None,
    *,
    centered: bool,
) -> tuple[SemanticPrefixTree, VectorSet | None, JsonObject]:
    if not params.project_dims or vectors.dim == 0:
        active = tree if tree is not None else build_tree(vectors, leaf_size=params.leaf_size)
        return active, None, {}
    if tree is not None:
        raise ValueError("an injected full-space tree cannot be used with project_dims")
    projected = prepare_projected_index(
        store,
        vectors,
        dims=params.project_dims,
        leaf_size=params.leaf_size,
        centered=centered,
    )
    return projected.tree, projected.vectors, projected.meta


def run_semantic_tiers(
    result: AuditResult,
    params: "AuditParams",
    docs: list[CorpusDoc],
    store: StoreView,
    goldset: list[GoldItem] | None,
    complete: LLMComplete | None,
    settled: set[tuple[str, str]],
    tree: SemanticPrefixTree | None,
) -> RunInputs:
    """Build/reuse the tree, run the semantic tier, then adjudicate if requested.

    Returns everything the bundle's own re-readings are taken from: the per-document chunk
    accounting the STAGE attribution needs, the per-document exclusion reasons behind the
    claim-token floor, and the ranked candidate list a budget replay is a prefix of. They are
    returned rather than recorded on the result because they are inputs to readings, not findings --
    and because the comparable set, the exclusion pass, and the ranking are all known exactly here
    and nowhere else.
    """
    governance = {doc.doc_id: doc.governance for doc in docs}
    centered = params.center_vectors and len(store.vectors) >= MIN_CENTERING_VECTORS
    if params.center_vectors and not centered:
        _LOG.info(
            "[conflicts] centering skipped: %d chunks is below the %d needed to estimate the "
            "corpus mean; comparing in the raw encoder space",
            len(store.vectors),
            MIN_CENTERING_VECTORS,
        )
    vectors = store.vectors.centered() if centered else store.vectors
    active, projected, projection_meta = _active_tree(
        params, store, vectors, tree, centered=centered
    )
    body_offsets = {doc.doc_id: doc.body_offset for doc in docs}
    selection = select_content_chunks(
        store.chunks, body_offsets, min_tokens=params.min_claim_tokens
    )
    allowed = selection.ordinals
    cos_threshold, null_payload = _calibrate_threshold(params, vectors, store.chunks, allowed)
    semantic_findings, pairs, semantic_stats = detect_semantic_pairs(
        active,
        vectors,
        store.chunks,
        governance,
        cos_threshold=cos_threshold,
        skip_doc_pairs=settled,
        body_offsets=body_offsets,
        min_tokens=params.min_claim_tokens,
        allowed=allowed,
        exclusion_counts=selection.stats(),
        projected_vectors=projected,
    )
    semantic_stats.extra.update(null_payload)
    result.tiers.append(semantic_stats)
    result.tree_meta = {
        **tree_meta(
            active,
            embedding_model=store.embedding_model,
            dim=store.dim,
            corpus_fingerprint=str(store.meta.get("corpus_fingerprint", "")),
            doc_fingerprints=store.doc_fingerprints,
            cos_threshold=cos_threshold,
        ),
        "centered": centered,
        **projection_meta,
        "cos_threshold": cos_threshold,
    }
    _record_needles(result, goldset, store, vectors, cos_threshold)
    _record_claims(result, params, store, governance, complete, semantic_findings, pairs)
    return RunInputs(
        chunks=DocumentChunks.of(store.chunks, allowed, settled),
        exclusions=DocumentExclusions.of(
            store.chunks, selection, min_claim_tokens=params.min_claim_tokens
        ),
        candidates=CandidateRecord.of(pairs, store.chunks, limit=_record_cap(params)),
    )


def _record_cap(params: "AuditParams") -> int:
    """How many document pairs this run's candidate record keeps.

    Three sources in order, and the order is the point. An explicit
    `--max-candidate-record-pairs` is what an operator who re-reads deeply sets, so it wins. Absent
    that, the run's own candidate budget is the cap: a record deeper than the budget answers about
    ranks the run itself refused to reach, and the re-read's question is downward. Absent both, the
    constant -- which is priced on the depth/cost curve rather than guessed.
    """
    return (
        params.max_candidate_record_pairs
        or params.max_candidate_pairs
        or DEFAULT_CANDIDATE_RECORD_PAIRS
    )


def _record_needles(
    result: AuditResult,
    goldset: list[GoldItem] | None,
    store: StoreView,
    vectors: VectorSet,
    cos_threshold: float,
) -> None:
    if not goldset:
        return
    _, report = analyze_needles(goldset, store.chunks, vectors, cos_threshold=cos_threshold)
    result.needles = report
    _LOG.info(
        "[conflicts] needles: %s of %s gold items are answerable from more than one document",
        report.get("ambiguous_items"),
        report.get("items"),
    )


def _record_claims(
    result: AuditResult,
    params: "AuditParams",
    store: StoreView,
    governance: dict[str, JsonObject],
    complete: LLMComplete | None,
    semantic_findings: list[Finding],
    pairs: list[tuple[int, int, float]],
) -> None:
    if TIER_CLAIM not in tiers_up_to(params.effort):
        result.findings.extend(semantic_findings)
        return
    if complete is None:
        raise SystemExit(
            "[conflicts] the claim tier needs a model endpoint: pass --conflict-model "
            "(and --conflict-backend) so candidate pairs can be adjudicated."
        )
    calibration = _calibrate(params, complete)
    cap = params.max_claim_pairs or len(pairs)
    selected = pairs[:cap]
    if len(selected) < len(pairs):
        _LOG.warning(
            "[conflicts] adjudicating %d of %d candidate pairs (--max-claim-pairs)",
            len(selected),
            len(pairs),
        )
    claim_findings, claim_stats, rows = adjudicate_pairs(
        selected, store.chunks, governance, complete
    )
    result.findings.extend(claim_findings)
    result.findings.extend(semantic_findings[len(selected) :])
    result.tiers.append(claim_stats)
    result.claim_precision = precision_block(rows, calibration, seed=params.null_seed)
    _LOG.info("[conflicts] tier=claim findings=%d", len(claim_findings))
    if not result.claim_precision["reported"]:
        _LOG.warning(
            "[conflicts] claim-tier precision not reported: %s", result.claim_precision["reason"]
        )


def _calibrate(params: "AuditParams", complete: LLMComplete) -> JsonObject | None:
    """Adjudicate the frozen probe first, so the precision block knows what it may print."""
    if not params.calibrate_adjudicator:
        return None
    probe = load_calibration_probe(params.calibration_probe)
    calibration = calibrate_adjudicator(probe, complete)
    _LOG.info(
        "[conflicts] adjudicator calibration: %s of %s frozen probe pairs agree (accuracy %s, "
        "Wilson 95%% lower bound %s, gate %s)",
        calibration["agreements"],
        calibration["parsed_pairs"],
        calibration["accuracy"],
        calibration["accuracy_wilson_95"][0],
        calibration["min_accuracy_lcb"],
    )
    return calibration
