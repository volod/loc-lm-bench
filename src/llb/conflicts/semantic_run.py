"""Semantic and claim-tier orchestration, split from the cheaper audit tiers."""

import logging
from typing import TYPE_CHECKING

from llb.conflicts.claim_tier import adjudicate_pairs
from llb.conflicts.constants import (
    DEFAULT_COSINE_THRESHOLD,
    MIN_CENTERING_VECTORS,
    TIER_CLAIM,
    tiers_up_to,
)
from llb.conflicts.corpus import CorpusDoc
from llb.conflicts.models import AuditResult, Finding
from llb.conflicts.needles import analyze_needles
from llb.conflicts.null_calibration import resolve_cos_threshold
from llb.conflicts.null_sampling import estimate_null_distribution
from llb.conflicts.projected_index import prepare_projected_index
from llb.conflicts.semantic_filter import select_content_chunks
from llb.conflicts.semantic_tier import build_tree, detect_semantic_pairs
from llb.conflicts.store_access import StoreView
from llb.conflicts.tree import SemanticPrefixTree
from llb.conflicts.tree_refresh import tree_meta
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.goldset.schema import GoldItem
from llb.prep.frontier_telemetry import LLMComplete

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
) -> None:
    """Build/reuse the tree, run the semantic tier, then adjudicate if requested."""
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
    cap = params.max_claim_pairs or len(pairs)
    selected = pairs[:cap]
    if len(selected) < len(pairs):
        _LOG.warning(
            "[conflicts] adjudicating %d of %d candidate pairs (--max-claim-pairs)",
            len(selected),
            len(pairs),
        )
    claim_findings, claim_stats = adjudicate_pairs(selected, store.chunks, governance, complete)
    result.findings.extend(claim_findings)
    result.findings.extend(semantic_findings[len(selected) :])
    result.tiers.append(claim_stats)
    _LOG.info("[conflicts] tier=claim findings=%d", len(claim_findings))
