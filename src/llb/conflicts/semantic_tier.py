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

import re
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
from llb.conflicts.tree import SemanticPrefixTree
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.rag.lexical import tokenize

EVIDENCE_COSINE = "cosine"


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


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def claim_token_count(text: str) -> int:
    """Content tokens in `text`, ignoring HTML comments (PDF page/provenance markers)."""
    return len(tokenize(_HTML_COMMENT.sub(" ", text)))


def content_ordinals(
    chunks: list[ChunkRecord],
    body_offsets: dict[str, int],
    *,
    min_tokens: int = MIN_CLAIM_TOKENS,
) -> set[int]:
    """Ordinals of chunks that carry a comparable CLAIM, excluding metadata and conversion residue.

    Two classes of chunk are dropped, both learned from real corpora rather than anticipated:

    Front matter -- every ingested document's governance block shares the same keys, so an
    archiving instruction and an appeals regulation match at cosine 0.9 on their `version:` and
    `language:` lines alone.

    Low-content chunks -- a converted PDF corpus is full of `<!-- source_pdf ... -->` markers,
    bare page numbers, and stub headings. Two such chunks match each other trivially, and on the
    quickstart HR corpus they were the single largest source of findings: the top-ranked
    "conflict" was one page marker against another.
    """
    return {
        ordinal
        for ordinal, chunk in enumerate(chunks)
        if int(chunk["char_end"]) > body_offsets.get(chunk["doc_id"], 0)
        and claim_token_count(chunk["text"]) >= min_tokens
    }


def cross_document_pairs(
    vectors: VectorSet,
    chunks: list[ChunkRecord],
    *,
    cos_threshold: float,
    skip_doc_pairs: Iterable[tuple[str, str]] = (),
    allowed: set[int] | None = None,
) -> list[tuple[int, int, float]]:
    """Matching chunk pairs that span two different, not-already-settled documents."""
    skip = set(skip_doc_pairs)
    out: list[tuple[int, int, float]] = []
    for left, right, similarity in vectors.pairs_above(cos_threshold):
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
) -> tuple[list[Finding], list[tuple[int, int, float]], TierStats]:
    """Provisional `duplicate` findings plus the raw pairs the claim tier adjudicates.

    The findings list is PARALLEL to the pairs list -- `findings[i]` is the provisional verdict
    for `pairs[i]`. Callers rely on that alignment to tell adjudicated pairs from unadjudicated
    ones by position, which is the only reliable way: the claim tier narrows a finding's span to
    the quoted claim, so its span key never equals the enclosing chunk's.
    """
    started = time.monotonic()
    stats = TierStats(tier=TIER_SEMANTIC)
    allowed = content_ordinals(chunks, body_offsets or {}, min_tokens=min_tokens)
    pairs = cross_document_pairs(
        vectors,
        chunks,
        cos_threshold=cos_threshold,
        skip_doc_pairs=skip_doc_pairs,
        allowed=allowed,
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
    stats.extra = {
        "cos_threshold": cos_threshold,
        "n_chunks": total,
        "comparable_chunks": len(allowed),
        "excluded_chunks": total - len(allowed),
        "min_claim_tokens": min_tokens,
        "exhaustive_pairs": total * (total - 1) // 2,
        "cross_document_pairs": len(pairs),
        "tree": tree.stats(),
    }
    return findings, pairs, stats
