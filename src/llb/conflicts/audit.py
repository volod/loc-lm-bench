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

from llb.conflicts.constants import (
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_LEAF_SIZE,
    MIN_CLAIM_TOKENS,
    TIER_HASH,
    TIER_LEXICAL,
    TIER_SEMANTIC,
    tiers_up_to,
)
from llb.conflicts.corpus import CorpusDoc, load_corpus_docs
from llb.conflicts.hash_tier import detect_hash_duplicates
from llb.conflicts.lexical_tier import detect_lexical_near_duplicates
from llb.conflicts.models import AuditResult, Finding, TierStats
from llb.conflicts.null_distribution import DEFAULT_NULL_SAMPLE_PAIRS, DEFAULT_NULL_SEED
from llb.conflicts.semantic_run import run_semantic_tiers
from llb.conflicts.store_access import StoreView
from llb.conflicts.tree import SemanticPrefixTree
from llb.core.contracts.common import JsonObject
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
    project_dims: int = 0

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
            "project_dims": self.project_dims,
        }


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
    run_semantic_tiers(result, params, docs, store, goldset, complete, settled, tree)
    return result
