"""The two model-free tiers, asserted against the planted fixture pairs.

Each planted pair must be found at the tier that is supposed to find it and NOT re-reported by a
later one, because the whole point of the effort dial is that a cheaper tier settling a pair saves
the more expensive tier from looking at it again.
"""

import pytest

from llb.conflicts.corpus import load_corpus_docs
from llb.conflicts.hash_tier import EVIDENCE_NORMALIZED, EVIDENCE_RAW, detect_hash_duplicates
from llb.conflicts.lexical_tier import (
    EVIDENCE_CONTAINMENT,
    candidate_pairs,
    containment,
    detect_lexical_near_duplicates,
    jaccard,
    shingles,
)
from llb.conflicts.constants import REL_DUPLICATE, REL_SUBSUMED_BY

from conflict_helpers import (
    DOC_2021,
    DOC_2021_COPY,
    DOC_2021_REFORMATTED,
    DOC_2024,
    DOC_ARCHIVE,
    DOC_EAPPEALS,
    FIXTURE_CORPUS,
)


@pytest.fixture
def docs():
    return load_corpus_docs(FIXTURE_CORPUS)


def _finding(findings, doc_a, doc_b):
    wanted = tuple(sorted([doc_a, doc_b]))
    matches = [f for f in findings if f.doc_pair() == wanted]
    assert matches, f"no finding for {wanted}"
    return matches[0]


def test_byte_identical_documents_are_raw_duplicates(docs):
    findings, _ = detect_hash_duplicates(docs)
    found = _finding(findings, DOC_2021, DOC_2021_COPY)
    assert (found.relation, found.evidence, found.score) == (REL_DUPLICATE, EVIDENCE_RAW, 1.0)


def test_reformatted_reissue_is_a_normalized_duplicate_with_an_edition_order(docs):
    """Case, whitespace, punctuation, and front matter differ; the content does not."""
    findings, _ = detect_hash_duplicates(docs)
    found = _finding(findings, DOC_2021_COPY, DOC_2021_REFORMATTED)
    assert (found.relation, found.evidence) == (REL_DUPLICATE, EVIDENCE_NORMALIZED)
    newer = found.b if found.staleness.newer_side == "b" else found.a
    assert newer.doc_id == DOC_2021_REFORMATTED
    assert found.staleness.basis == "effective_date"


def test_settled_pairs_are_the_full_group_closure_not_just_the_reported_chain(docs):
    """Duplication is transitive: all three copies pair with each other, however few we report."""
    findings, settled = detect_hash_duplicates(docs)
    group = {DOC_2021, DOC_2021_COPY, DOC_2021_REFORMATTED}
    expected = {tuple(sorted([a, b])) for a in group for b in group if a < b}
    assert expected <= settled
    assert len(findings) < len(expected)


def test_unrelated_document_is_never_a_hash_duplicate(docs):
    findings, _ = detect_hash_duplicates(docs)
    assert all(DOC_ARCHIVE not in finding.doc_pair() for finding in findings)


def test_absorbed_note_is_reported_as_subsumed(docs):
    """The note's content sits whole inside the 2024 article, so the note is subsumed by it."""
    findings, _ = detect_lexical_near_duplicates(docs)
    found = _finding(findings, DOC_EAPPEALS, DOC_2024)
    assert (found.relation, found.evidence) == (REL_SUBSUMED_BY, EVIDENCE_CONTAINMENT)
    assert found.a.doc_id == DOC_EAPPEALS, "side a must be the subsumed document"
    assert found.b.doc_id == DOC_2024


def test_subsumption_is_the_low_jaccard_case_lsh_blocking_would_miss(docs):
    """Regression guard for the blocking strategy: containment is high while Jaccard is low."""
    by_id = {doc.doc_id: shingles(doc.body) for doc in docs}
    note, article = by_id[DOC_EAPPEALS], by_id[DOC_2024]
    assert containment(note, article) >= 0.9
    assert jaccard(note, article) < 0.4
    assert (
        tuple(sorted([0, 1]))
        in candidate_pairs([note, article])  # still blocked together despite low Jaccard
    )


def test_lexical_tier_does_not_re_report_hash_settled_pairs(docs):
    hash_findings, settled = detect_hash_duplicates(docs)
    findings, _ = detect_lexical_near_duplicates(docs, skip_doc_pairs=settled)
    assert all(finding.doc_pair() not in settled for finding in findings)
    assert len(findings) == 1, "only the subsumption is new at this tier"


def test_revision_is_left_for_the_claim_tier(docs):
    """2021 vs 2024 is neither near-identical nor contained, so no lexical verdict is claimed."""
    _, settled = detect_hash_duplicates(docs)
    findings, _ = detect_lexical_near_duplicates(docs, skip_doc_pairs=settled)
    assert tuple(sorted([DOC_2021, DOC_2024])) not in {f.doc_pair() for f in findings}


def test_lexical_output_is_deterministic(docs):
    _, settled = detect_hash_duplicates(docs)
    first, _ = detect_lexical_near_duplicates(docs, skip_doc_pairs=settled)
    second, _ = detect_lexical_near_duplicates(docs, skip_doc_pairs=settled)
    assert [f.payload() for f in first] == [f.payload() for f in second]


def test_boilerplate_shingles_do_not_block_every_pair():
    """A shingle shared by nearly every document carries no evidence and is skipped."""
    shared = {1, 2, 3}
    sets = [shared | {10 + i} for i in range(10)]
    assert candidate_pairs(sets, max_doc_frequency=0.5) == set()


def test_governance_only_change_still_reads_as_duplicate_content(docs):
    """corpus_doc_fingerprints folds governance in; content hashing must not."""
    copy = next(doc for doc in docs if doc.doc_id == DOC_2021_COPY)
    reformatted = next(doc for doc in docs if doc.doc_id == DOC_2021_REFORMATTED)
    assert copy.raw_sha != reformatted.raw_sha
    assert copy.normalized_sha == reformatted.normalized_sha
