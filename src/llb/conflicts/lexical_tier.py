"""Tier 2 (`lexical`): near-duplicate and subsumed documents via shingle overlap.

Documents become sets of word 5-gram shingles over the Ukrainian-normalized token stream, so
casing, punctuation, and apostrophe variants never split a shingle.

Two measures, two relations:
  - Jaccard >= threshold        -> `duplicate` (the documents are mutually near-identical)
  - containment(A in B) >= t    -> `subsumed_by` for A (B says everything A says, and more)

Containment is what catches the note whose content a larger article absorbed whole. That case is
also why blocking here is an inverted shingle index rather than MinHash/LSH: a short document
inside a long one has LOW Jaccard by construction (a 25-shingle note inside a 130-shingle article
scores about 0.19), and banded LSH is tuned to find high-Jaccard pairs -- it would miss exactly
the subsumption this tier exists to report. The inverted index instead yields every pair sharing
at least one discriminative shingle, and the exact Jaccard and containment are computed on those
pairs' full shingle sets, so no candidate is lost to a probabilistic sketch.
"""

import time
from collections.abc import Iterable

from llb.conflicts.constants import (
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_JACCARD_THRESHOLD,
    MAX_SHINGLE_DOC_FREQUENCY,
    REL_DUPLICATE,
    REL_SUBSUMED_BY,
    SHINGLE_SIZE,
    TIER_LEXICAL,
)
from llb.conflicts.corpus import CorpusDoc, whole_doc_span
from llb.conflicts.governance import compare_editions
from llb.conflicts.hashing import stable_hash64
from llb.conflicts.models import ClaimRef, Finding, TierStats
from llb.rag.lexical import tokenize

EVIDENCE_JACCARD = "jaccard"
EVIDENCE_CONTAINMENT = "containment"


def shingles(text: str, width: int = SHINGLE_SIZE) -> set[int]:
    """Hashed word n-gram shingles of `text`. Documents shorter than `width` hash whole."""
    tokens = tokenize(text)
    if not tokens:
        return set()
    if len(tokens) < width:
        return {stable_hash64(" ".join(tokens))}
    return {stable_hash64(" ".join(tokens[i : i + width])) for i in range(len(tokens) - width + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    """Symmetric overlap: shared shingles over the union."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)


def containment(inner: set[int], outer: set[int]) -> float:
    """Fraction of `inner`'s shingles that also occur in `outer`."""
    return len(inner & outer) / len(inner) if inner else 0.0


def candidate_pairs(
    doc_shingles: list[set[int]], max_doc_frequency: float = MAX_SHINGLE_DOC_FREQUENCY
) -> set[tuple[int, int]]:
    """Every document pair sharing at least one discriminative shingle.

    Shingles occurring in more than `max_doc_frequency` of the corpus are skipped: they are
    boilerplate (shared headers, standard legal formulas), so they pair nearly every document
    with every other while carrying no evidence that two documents say the same thing.
    """
    inverted: dict[int, list[int]] = {}
    for index, shingle_set in enumerate(doc_shingles):
        for shingle in shingle_set:
            inverted.setdefault(shingle, []).append(index)
    limit = max(2, int(max_doc_frequency * len(doc_shingles)))
    pairs: set[tuple[int, int]] = set()
    for postings in inverted.values():
        if len(postings) > limit:
            continue
        for position, left in enumerate(postings):
            for right in postings[position + 1 :]:
                pairs.add((left, right))
    return pairs


def _claim_ref(doc: CorpusDoc) -> ClaimRef:
    start, end = whole_doc_span(doc)
    return ClaimRef(
        doc_id=doc.doc_id,
        char_start=start,
        char_end=end,
        text=doc.text,
        governance=doc.governance,
    )


def _duplicate_finding(left: CorpusDoc, right: CorpusDoc, score: float) -> Finding:
    a, b = _claim_ref(left), _claim_ref(right)
    return Finding(
        relation=REL_DUPLICATE,
        tier=TIER_LEXICAL,
        a=a,
        b=b,
        score=score,
        evidence=EVIDENCE_JACCARD,
        staleness=compare_editions(a.governance, b.governance),
        rationale=f"word-{SHINGLE_SIZE}-gram Jaccard {score:.3f}",
    )


def _subsumption_finding(inner: CorpusDoc, outer: CorpusDoc, score: float) -> Finding:
    """`inner` is subsumed by `outer`: side a is always the subsumed (less specific) document."""
    a, b = _claim_ref(inner), _claim_ref(outer)
    return Finding(
        relation=REL_SUBSUMED_BY,
        tier=TIER_LEXICAL,
        a=a,
        b=b,
        score=score,
        evidence=EVIDENCE_CONTAINMENT,
        staleness=compare_editions(a.governance, b.governance),
        rationale=(
            f"{score:.3f} of {inner.doc_id}'s shingles occur in {outer.doc_id}, "
            f"which is {len(outer.text) - len(inner.text)} characters longer"
        ),
    )


def detect_lexical_near_duplicates(
    docs: list[CorpusDoc],
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    skip_doc_pairs: Iterable[tuple[str, str]] = (),
) -> tuple[list[Finding], TierStats]:
    """Near-duplicate and subsumed documents. `skip_doc_pairs` are already-settled doc pairs."""
    started = time.monotonic()
    stats = TierStats(tier=TIER_LEXICAL)
    skip = set(skip_doc_pairs)
    if len(docs) < 2:
        stats.seconds = time.monotonic() - started
        return [], stats

    doc_shingles = [shingles(doc.body) for doc in docs]
    candidates = sorted(candidate_pairs(doc_shingles))
    stats.candidate_pairs = len(candidates)

    findings: list[Finding] = []
    for left, right in candidates:
        a_doc, b_doc = docs[left], docs[right]
        if tuple(sorted([a_doc.doc_id, b_doc.doc_id])) in skip:
            continue
        a_set, b_set = doc_shingles[left], doc_shingles[right]
        overlap = jaccard(a_set, b_set)
        if overlap >= jaccard_threshold:
            findings.append(_duplicate_finding(a_doc, b_doc, overlap))
            continue
        # Not mutually near-identical: does the smaller document sit inside the larger one?
        inner, outer, inner_set, outer_set = (
            (a_doc, b_doc, a_set, b_set)
            if len(a_set) <= len(b_set)
            else (b_doc, a_doc, b_set, a_set)
        )
        covered = containment(inner_set, outer_set)
        if covered >= containment_threshold and len(inner_set) < len(outer_set):
            findings.append(_subsumption_finding(inner, outer, covered))
    stats.findings = len(findings)
    stats.seconds = time.monotonic() - started
    stats.extra = {
        "jaccard_threshold": jaccard_threshold,
        "containment_threshold": containment_threshold,
        "shingle_size": SHINGLE_SIZE,
    }
    return findings, stats
