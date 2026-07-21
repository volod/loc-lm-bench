"""Tier 3 (`semantic`): claim-level candidate groups from the semantic prefix tree.

Works on the store's CHUNK vectors, so a finding points at the span that actually carries the
claim rather than at a whole document. Chunk pairs from the same document are dropped -- a
document restating itself is a drafting problem, not a corpus-conflict one -- and so are pairs
whose documents the hash tier already settled as byte-identical.

A pair at or above the cosine threshold is reported as `duplicate` when nothing further runs. That
is deliberately provisional: cosine says "these two spans are about the same thing", not "these
two spans agree". Distinguishing agreement from contradiction needs the claim tier, and a
`semantic`-tier run says so in its evidence field.
"""

import time
from collections.abc import Iterable

from llb.conflicts.constants import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_LEAF_SIZE,
    MIN_CLAIM_TOKENS,
    REL_DUPLICATE,
    TIER_SEMANTIC,
)
from llb.conflicts.governance import compare_editions
from llb.conflicts.models import ClaimRef, Finding, TierStats
from llb.conflicts.projection import euclidean_threshold
from llb.conflicts.projected_index import exact_projected_pairs
from llb.conflicts.semantic_filter import claim_token_count as _claim_token_count
from llb.conflicts.semantic_filter import select_content_chunks
from llb.conflicts.tree import SemanticPrefixTree
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord

EVIDENCE_COSINE = "cosine"


def claim_token_count(text: str) -> int:
    """Content tokens in text, ignoring HTML comments (compatibility helper)."""
    return _claim_token_count(text)


def chunk_claim_ref(chunk: ChunkRecord, governance: JsonObject) -> ClaimRef:
    """A claim reference for a whole chunk span (exact offsets, verbatim corpus text)."""
    return ClaimRef(
        doc_id=chunk["doc_id"],
        char_start=int(chunk["char_start"]),
        char_end=int(chunk["char_end"]),
        text=chunk["text"],
        chunk_id=chunk.get("chunk_id"),
        governance=governance,
    )


def content_ordinals(
    chunks: list[ChunkRecord],
    body_offsets: dict[str, int],
    *,
    min_tokens: int = MIN_CLAIM_TOKENS,
) -> set[int]:
    """Ordinals of chunks that carry a comparable CLAIM, excluding metadata and conversion residue.

    Three classes of chunk are dropped, all learned from real corpora rather than anticipated:

    Front matter -- every ingested document's governance block shares the same keys, so an
    archiving instruction and an appeals regulation match at cosine 0.9 on their `version:` and
    `language:` lines alone.

    Low-content chunks -- a converted PDF corpus is full of `<!-- source_pdf ... -->` markers,
    bare page numbers, and stub headings. Two such chunks match each other trivially, and on the
    quickstart HR corpus they were the single largest source of findings: the top-ranked
    "conflict" was one page marker against another.

    Repeated metadata blocks -- claim-sized publication/registry records under the same structural
    heading in multiple documents, confirmed from shared tokens and numeric-field density.
    """
    return select_content_chunks(chunks, body_offsets, min_tokens=min_tokens).ordinals


def cross_document_pairs(
    vectors: VectorSet,
    chunks: list[ChunkRecord],
    *,
    cos_threshold: float,
    skip_doc_pairs: Iterable[tuple[str, str]] = (),
    allowed: set[int] | None = None,
    candidates: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float]]:
    """Matching chunk pairs that span two different, not-already-settled documents."""
    skip = set(skip_doc_pairs)
    out: list[tuple[int, int, float]] = []
    matches = (
        vectors.pairs_above(cos_threshold)
        if candidates is None
        else vectors.pairs_above_candidates(candidates, cos_threshold)
    )
    for left, right, similarity in matches:
        if allowed is not None and (left not in allowed or right not in allowed):
            continue
        left_doc, right_doc = chunks[left]["doc_id"], chunks[right]["doc_id"]
        if left_doc == right_doc:
            continue
        if tuple(sorted([left_doc, right_doc])) in skip:
            continue
        out.append((left, right, similarity))
    return out


def build_tree(vectors: VectorSet, *, leaf_size: int = DEFAULT_LEAF_SIZE) -> SemanticPrefixTree:
    """Build the semantic prefix tree over already-loaded store vectors."""
    return SemanticPrefixTree.build(vectors, leaf_size=leaf_size)


def detect_semantic_pairs(
    tree: SemanticPrefixTree,
    vectors: VectorSet,
    chunks: list[ChunkRecord],
    governance_by_doc: dict[str, JsonObject],
    *,
    cos_threshold: float = DEFAULT_COSINE_THRESHOLD,
    skip_doc_pairs: Iterable[tuple[str, str]] = (),
    body_offsets: dict[str, int] | None = None,
    min_tokens: int = MIN_CLAIM_TOKENS,
    allowed: set[int] | None = None,
    exclusion_counts: JsonObject | None = None,
    projected_vectors: VectorSet | None = None,
) -> tuple[list[Finding], list[tuple[int, int, float]], TierStats]:
    """Provisional `duplicate` findings plus the raw pairs the claim tier adjudicates.

    The findings list is PARALLEL to the pairs list -- `findings[i]` is the provisional verdict
    for `pairs[i]`. Callers rely on that alignment to tell adjudicated pairs from unadjudicated
    ones by position, which is the only reliable way: the claim tier narrows a finding's span to
    the quoted claim, so its span key never equals the enclosing chunk's.
    """
    started = time.monotonic()
    stats = TierStats(tier=TIER_SEMANTIC)
    # The caller may pass the comparable set it already computed (the audit needs it to sample
    # the null distribution over exactly these pairs); otherwise derive it here.
    if allowed is None:
        selection = select_content_chunks(chunks, body_offsets or {}, min_tokens=min_tokens)
        allowed = selection.ordinals
        exclusion_counts = selection.stats()
    candidates = None
    projected_backend = None
    project_dims = None
    if projected_vectors is not None:
        project_dims = projected_vectors.dim
        candidates, projected_backend = exact_projected_pairs(
            tree,
            projected_vectors,
            euclidean_threshold(cos_threshold),
        )
    pairs = cross_document_pairs(
        vectors,
        chunks,
        cos_threshold=cos_threshold,
        skip_doc_pairs=skip_doc_pairs,
        allowed=allowed,
        candidates=candidates,
    )
    findings: list[Finding] = []
    for left, right, similarity in pairs:
        a = chunk_claim_ref(chunks[left], governance_by_doc.get(chunks[left]["doc_id"], {}))
        b = chunk_claim_ref(chunks[right], governance_by_doc.get(chunks[right]["doc_id"], {}))
        findings.append(
            Finding(
                relation=REL_DUPLICATE,
                tier=TIER_SEMANTIC,
                a=a,
                b=b,
                score=similarity,
                evidence=EVIDENCE_COSINE,
                staleness=compare_editions(a.governance, b.governance),
                rationale=(
                    f"chunk cosine {similarity:.3f} >= {cos_threshold}; same topic, agreement "
                    "not yet adjudicated (run --effort claim to label the relation)"
                ),
            )
        )
    total = len(chunks)
    stats.candidate_pairs = len(pairs)
    stats.findings = len(findings)
    stats.seconds = time.monotonic() - started
    exhaustive = total * (total - 1) // 2
    stats.extra = {
        "cos_threshold": cos_threshold,
        "n_chunks": total,
        "comparable_chunks": len(allowed),
        "excluded_chunks": total - len(allowed),
        "min_claim_tokens": min_tokens,
        "exhaustive_pairs": exhaustive,
        "cross_document_pairs": len(pairs),
        "tree": tree.stats(),
        **(exclusion_counts or {}),
    }
    if candidates is not None:
        stats.extra.update(
            {
                "blocking": "pca-euclidean",
                "projected_backend": projected_backend,
                "project_dims": project_dims,
                "projected_candidate_pairs": len(candidates),
                "projected_pruned_pairs": exhaustive - len(candidates),
                "projected_pruning_fraction": (
                    (exhaustive - len(candidates)) / exhaustive if exhaustive else 0.0
                ),
                "full_space_comparisons": len(candidates),
            }
        )
    return findings, pairs, stats
