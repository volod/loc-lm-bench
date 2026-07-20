"""Run the effort tiers in order and assemble one audit result.

Tiers are cumulative and each one narrows the work for the next: documents the hash tier proves
byte-identical are not re-examined lexically or semantically, and only the semantic tier's
surviving chunk pairs are ever sent to a model. That is the whole point of the effort dial --
`hash` is a single pass over the corpus, `claim` is a model call per surviving candidate, and the
tiers in between exist so that number stays small.

Detection only: nothing here edits, deletes, or reorders a single corpus byte.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from llb.conflicts.claim_tier import adjudicate_pairs
from llb.conflicts.constants import (
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_LEAF_SIZE,
    MIN_CENTERING_VECTORS,
    MIN_CLAIM_TOKENS,
    TIER_CLAIM,
    TIER_HASH,
    TIER_LEXICAL,
    TIER_SEMANTIC,
    tiers_up_to,
)
from llb.conflicts.corpus import CorpusDoc, load_corpus_docs
from llb.conflicts.hash_tier import detect_hash_duplicates
from llb.conflicts.lexical_tier import detect_lexical_near_duplicates
from llb.conflicts.models import AuditResult, Finding, TierStats
from llb.conflicts.needles import analyze_needles
from llb.conflicts.null_calibration import resolve_cos_threshold
from llb.conflicts.null_distribution import DEFAULT_NULL_SAMPLE_PAIRS, DEFAULT_NULL_SEED
from llb.conflicts.null_sampling import estimate_null_distribution
from llb.conflicts.semantic_tier import build_tree, content_ordinals, detect_semantic_pairs
from llb.conflicts.store_access import StoreView
from llb.conflicts.tree import SemanticPrefixTree
from llb.conflicts.tree_refresh import tree_meta
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.goldset.schema import GoldItem
from llb.prep.frontier_telemetry import LLMComplete

_LOG = logging.getLogger(__name__)


@dataclass
class AuditParams:
    """Thresholds and knobs for one audit run."""

    effort: str
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD
    cos_threshold: float | None = None
    cos_quantile: float | None = None
    max_candidate_pairs: int | None = None
    null_sample_pairs: int = DEFAULT_NULL_SAMPLE_PAIRS
    null_seed: int = DEFAULT_NULL_SEED
    leaf_size: int = DEFAULT_LEAF_SIZE
    max_claim_pairs: int = 0  # 0 == adjudicate every candidate pair
    min_claim_tokens: int = MIN_CLAIM_TOKENS
    center_vectors: bool = True

    def payload(self) -> JsonObject:
        return {
            "effort": self.effort,
            "jaccard_threshold": self.jaccard_threshold,
            "containment_threshold": self.containment_threshold,
            "cos_threshold": self.cos_threshold,
            "cos_quantile": self.cos_quantile,
            "max_candidate_pairs": self.max_candidate_pairs,
            "null_sample_pairs": self.null_sample_pairs,
            "null_seed": self.null_seed,
            "leaf_size": self.leaf_size,
            "max_claim_pairs": self.max_claim_pairs,
            "min_claim_tokens": self.min_claim_tokens,
            "center_vectors": self.center_vectors,
        }


def _governance_by_doc(docs: list[CorpusDoc]) -> dict[str, JsonObject]:
    return {doc.doc_id: doc.governance for doc in docs}


def _hash_stats(findings: list[Finding], docs: list[CorpusDoc], seconds: float) -> TierStats:
    return TierStats(
        tier=TIER_HASH,
        candidate_pairs=len(docs),
        findings=len(findings),
        seconds=seconds,
        extra={"n_docs": len(docs)},
    )


def run_audit(
    corpus_root: Path | str,
    params: AuditParams,
    *,
    store: StoreView | None = None,
    goldset: list[GoldItem] | None = None,
    complete: LLMComplete | None = None,
    tree: SemanticPrefixTree | None = None,
) -> AuditResult:
    """Run every tier up to `params.effort` and return the assembled result.

    `store` is required from the `semantic` tier upward; `complete` from `claim` upward. `tree`
    reuses an already-built (or refreshed) semantic prefix tree instead of building a fresh one.
    """
    tiers = tiers_up_to(params.effort)
    docs = load_corpus_docs(corpus_root)
    result = AuditResult(
        effort=params.effort,
        corpus_root=str(corpus_root),
        n_docs=len(docs),
        params=params.payload(),
    )

    started = time.monotonic()
    hash_findings, settled = detect_hash_duplicates(docs)
    result.findings.extend(hash_findings)
    result.tiers.append(_hash_stats(hash_findings, docs, time.monotonic() - started))
    _LOG.info("[conflicts] tier=hash docs=%d findings=%d", len(docs), len(hash_findings))

    if TIER_LEXICAL in tiers:
        lexical_findings, lexical_stats = detect_lexical_near_duplicates(
            docs,
            jaccard_threshold=params.jaccard_threshold,
            containment_threshold=params.containment_threshold,
            skip_doc_pairs=settled,
        )
        result.findings.extend(lexical_findings)
        result.tiers.append(lexical_stats)
        _LOG.info("[conflicts] tier=lexical findings=%d", len(lexical_findings))

    if TIER_SEMANTIC not in tiers:
        return result
    if store is None:
        raise SystemExit(
            "[conflicts] the semantic tier needs a built store: pass --store or build one "
            "with `make build-index`."
        )
    _run_semantic_tiers(result, params, docs, store, goldset, complete, settled, tree)
    return result


def _calibrate_threshold(
    params: AuditParams,
    vectors: "VectorSet",
    chunks: list[ChunkRecord],
    allowed: set[int],
) -> tuple[float, JsonObject]:
    """Resolve the operating cosine and the null-distribution record that justifies it.

    Sampling only runs when a quantile was asked for: it costs a pass over the sampled pairs, and
    a run that names an absolute threshold has nothing to calibrate.
    """
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
    payload: dict[str, object] = {
        "cos_threshold": threshold,
        "cos_threshold_source": source,
    }
    if distribution is not None:
        payload["null_distribution"] = distribution.payload(
            resolved_quantile, params.max_candidate_pairs
        )
    return threshold, payload


def _run_semantic_tiers(
    result: AuditResult,
    params: AuditParams,
    docs: list[CorpusDoc],
    store: StoreView,
    goldset: list[GoldItem] | None,
    complete: LLMComplete | None,
    settled: set[tuple[str, str]],
    tree: SemanticPrefixTree | None,
) -> None:
    """Build/reuse the tree, run the semantic tier, then adjudicate if the claim tier is on."""
    governance = _governance_by_doc(docs)
    # Encoder spaces are anisotropic, so an uncentered threshold sits barely above the similarity
    # of two unrelated chunks. Centering is what makes `cos_threshold` mean the same thing on a
    # real corpus that it means on a toy one -- but only once there are enough vectors for the
    # mean to be an estimate rather than an accident of which few documents are present.
    centered = params.center_vectors and len(store.vectors) >= MIN_CENTERING_VECTORS
    if params.center_vectors and not centered:
        _LOG.info(
            "[conflicts] centering skipped: %d chunks is below the %d needed to estimate the "
            "corpus mean; comparing in the raw encoder space",
            len(store.vectors),
            MIN_CENTERING_VECTORS,
        )
    vectors = store.vectors.centered() if centered else store.vectors
    active = tree if tree is not None else build_tree(vectors, leaf_size=params.leaf_size)
    body_offsets = {doc.doc_id: doc.body_offset for doc in docs}
    # The comparable set is computed here rather than left to the tier, because the null
    # distribution must be sampled from exactly the pairs the scan will consider: front matter and
    # PDF residue score high against each other and would inflate the estimated quantile.
    allowed = content_ordinals(store.chunks, body_offsets, min_tokens=params.min_claim_tokens)
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
    }

    if goldset:
        _, needle_report = analyze_needles(
            goldset, store.chunks, vectors, cos_threshold=cos_threshold
        )
        result.needles = needle_report
        _LOG.info(
            "[conflicts] needles: %s of %s gold items are answerable from more than one document",
            needle_report.get("ambiguous_items"),
            needle_report.get("items"),
        )

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
    # `semantic_findings` is parallel to `pairs`, so the tail beyond the cap is exactly the set of
    # pairs the model never saw. Those keep their provisional semantic finding, which is why a
    # capped run reports the same pairs as an uncapped one -- just fewer of them adjudicated.
    # Matching on span keys instead would double-report: the claim tier narrows a finding to the
    # quoted claim, so its span never equals the enclosing chunk's.
    result.findings.extend(semantic_findings[len(selected) :])
    result.tiers.append(claim_stats)
    _LOG.info("[conflicts] tier=claim findings=%d", len(claim_findings))
