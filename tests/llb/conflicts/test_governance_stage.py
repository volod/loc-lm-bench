"""Which STAGE a run lost an orderable document pair at -- the knob, not the list of knobs.

The coverage counts already say a corpus could have supplied an orderable pair and the run did not
return one. That reading ends in four knobs at once (raise `--effort`, raise
`--max-candidate-pairs`, re-chunk, re-embed) because nothing in the run recorded which of them
dropped the pair. These tests pin the attribution that does:

1. the pair itself -- one lost orderable document pair, drawn from the EARLIEST stage the run lost
   one at and taken in corpus order within that stage, with returned pairs and unorderable pairs
   skipped;
2. the stage -- read off the chunk accounting the semantic tier produced anyway, at each of the
   five places a pair can stop: the effort dial, duplicate collapse, chunking, the claim-token
   floor, and candidate selection;
3. the cost -- the earliest stage is found from per-DOCUMENT facts, so a corpus whose first lost
   pair sits far into the pair space is still scanned linearly;
4. the silence -- a run that returned every orderable document pair it could have prints no
   attribution at all, because there is no knob to turn.

The end-to-end cases run the real audit over throwaway corpora and a StoreView of real chunk
records with the hashed bag-of-words fake, so the stage is decided by the pipeline rather than by
a hand-built payload.
"""

from pathlib import Path

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import TIER_HASH, TIER_SEMANTIC
from llb.conflicts.document_chunks import DocumentChunks
from llb.conflicts.governance_coverage import RETRIEVAL_KNOBS, coverage_reading
from llb.conflicts.governance_stage import (
    LOST_PAIR_FIELD,
    STAGE_CANDIDATES,
    STAGE_CHUNKING,
    STAGE_CLAIM_FLOOR,
    STAGE_DUPLICATE_COLLAPSE,
    STAGE_EFFORT,
    STAGE_KNOBS,
    earliest_lost_orderable_pair,
    first_lost_orderable_pair,
    lost_pair_attribution,
    lost_pair_sentence,
    returned_doc_pairs,
)
from llb.conflicts.store_access import StoreView
from llb.conflicts.vectorops import VectorSet

from conflict_helpers import FAKE_COS_THRESHOLD, bow_vector, chunk_fixture

SHORT = "коротка примітка"
# Enough tokens to clear `MIN_CLAIM_TOKENS`, and few enough to chunk as one heading-sized unit.
BODY_TOKENS = 40
# The bodies below share tokens only where a test wants two documents to LOOK alike: the fake
# embedder is a hashed bag of words, so token overlap is the whole similarity signal, and two
# bodies drawn from disjoint windows are unrelated to the lexical tier and to cosine alike.
CORE = 4

_2021 = {"effective_date": "2021-01-01"}
_2024 = {"effective_date": "2024-01-01"}
_2026 = {"effective_date": "2026-01-01"}


def _body(start: int, count: int = BODY_TOKENS) -> str:
    return " ".join(f"пункт{index:04d}" for index in range(start, start + count))


def _corpus(root: Path, documents: dict[str, tuple[str, str]]) -> Path:
    """`{name: (effective_date, body)}` written as a corpus of dated Markdown documents."""
    root.mkdir(parents=True, exist_ok=True)
    for name, (date, body) in documents.items():
        (root / name).write_text(f"---\neffective_date: {date}\n---\n{body}\n", encoding="utf-8")
    return root


def _store(root: Path, *, without: str = "") -> StoreView:
    """A StoreView over the corpus's real chunks, optionally missing one document's entirely."""
    chunks = [chunk for chunk in chunk_fixture(root) if chunk["doc_id"] != without]
    return StoreView(
        index_dir=root,
        chunks=chunks,
        vectors=VectorSet([bow_vector(chunk["text"]) for chunk in chunks]),
        meta={"embedding_model": "fake-hashed-bow", "corpus_fingerprint": "stage"},
    )


def _lost(result) -> dict:
    return result.governance_coverage[LOST_PAIR_FIELD]


# --- the pair: which one is named ------------------------------------------------------------


def test_the_first_orderable_pair_no_returned_row_joins_is_the_one_named():
    """Corpus order, with returned pairs skipped -- a pair that survived is not a pair to explain."""
    documents = [("a.md", _2021), ("b.md", _2021), ("c.md", _2024)]

    assert first_lost_orderable_pair(documents, set()) == ("a.md", "c.md")
    assert first_lost_orderable_pair(documents, {("a.md", "c.md")}) == ("b.md", "c.md")
    assert first_lost_orderable_pair(documents, {("a.md", "c.md"), ("b.md", "c.md")}) is None


def test_a_pair_the_corpus_cannot_order_is_never_the_pair_named():
    """The same orderability test the counts use: one date, or a date on one side only, is none."""
    assert first_lost_orderable_pair([("a.md", _2021), ("b.md", _2021)], set()) is None
    assert first_lost_orderable_pair([("a.md", _2021), ("b.md", {})], set()) is None
    assert first_lost_orderable_pair([("a.md", {}), ("b.md", {})], set()) is None
    assert first_lost_orderable_pair([("a.md", _2021), ("b.md", _2024)], set()) == ("a.md", "b.md")


def test_the_returned_pairs_are_read_off_the_rows_unordered():
    """A row names its two sides in adjudication order; a document pair has no order at all."""
    rows = [{"a": {"doc_id": "z.md"}, "b": {"doc_id": "a.md"}}]

    assert returned_doc_pairs(rows) == {("a.md", "z.md")}


def test_the_earliest_stage_is_named_even_when_another_pair_sorts_first():
    """The rule: corpus order picks WITHIN a stage, never between two of them.

    `a.md` + `b.md` sorts first and is a candidate-selection miss -- two documents that were never
    going to pair. `z.md` is in no chunk of the store at all, which is a gap an operator can fix,
    and it is the pair that names it that has to be reported.
    """
    documents = [("a.md", _2021), ("b.md", _2024), ("z.md", _2026)]
    chunks = DocumentChunks(stored={"a.md": 2, "b.md": 2}, comparable={"a.md": 2, "b.md": 2})

    assert first_lost_orderable_pair(documents, set()) == ("a.md", "b.md")
    assert earliest_lost_orderable_pair(documents, set(), chunks) == ("a.md", "z.md")


def test_the_stage_with_no_knob_never_outranks_a_stage_with_one():
    """Duplicate collapse is a non-loss: the claim is in the store under the copy that survived.

    So it sorts last however early in the pipeline it sits, and is reported only when a run lost
    nothing else -- otherwise a report whose one advice is "none" would hide a real knob.
    """
    documents = [("a.md", _2021), ("copy.md", _2024), ("short.md", _2026)]
    stored = {"a.md": 2, "kept.md": 2, "short.md": 1}
    copies = {"copy.md": ("kept.md",), "kept.md": ("copy.md",)}
    chunks = DocumentChunks(stored=stored, comparable={"a.md": 2, "kept.md": 2}, copies=copies)

    assert earliest_lost_orderable_pair(documents, set(), chunks) == ("a.md", "short.md")
    # With that floor miss returned, the collapsed pairs are all that is left -- and now they print.
    returned = {("a.md", "short.md")}
    assert earliest_lost_orderable_pair(documents, returned, chunks) == ("a.md", "copy.md")


class _CountingPairs(set):
    """A returned-pair set that counts how many document pairs the scan actually tests."""

    lookups = 0

    def __contains__(self, item: object) -> bool:
        self.lookups += 1
        return super().__contains__(item)


def test_the_earliest_stage_is_found_without_walking_the_pair_space():
    """The cost bound: every stage below `candidates` is a property of a DOCUMENT, not of a pair.

    Sixty dated documents whose only lost pair is the last one in corpus order. The corpus-order
    scan has to reach it through the whole pair space; the earliest-stage scan finds the one
    document with no chunk in the store first and tests only ITS pairs.
    """
    size = 60
    documents = [
        (f"d{index:02d}.md", {"effective_date": f"20{index:02d}-01-01"}) for index in range(size)
    ]
    names = [doc_id for doc_id, _ in documents]
    returned = {
        tuple(sorted([left, right]))
        for left in names
        for right in names
        if left < right and not (right == names[-1] and left == names[-2])
    }
    chunks = DocumentChunks(
        stored={doc_id: 1 for doc_id in names[:-1]},
        comparable={doc_id: 1 for doc_id in names[:-1]},
    )

    earliest_pairs, corpus_order_pairs = _CountingPairs(returned), _CountingPairs(returned)
    assert earliest_lost_orderable_pair(documents, earliest_pairs, chunks) == (
        names[-2],
        names[-1],
    )
    assert first_lost_orderable_pair(documents, corpus_order_pairs) == (names[-2], names[-1])

    assert earliest_pairs.lookups <= size, "one pass over the corpus, for the one lost document"
    assert corpus_order_pairs.lookups > size * (size - 1) // 4, "the corpus-order scan pays pairs"


def test_nothing_orderable_in_the_corpus_means_no_attribution_and_no_scan():
    """The guard the counts already answer: a corpus that can order nothing has nothing to lose."""
    coverage = {"orderable_document_pairs": 0}

    assert lost_pair_attribution(coverage, [("a.md", _2021)], [], None) == {}
    assert lost_pair_sentence({}) == ""


# --- the stage: which knob it names ----------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "stage", "named"),
    [
        (None, STAGE_EFFORT, "no store"),
        (DocumentChunks(stored={"a.md": 3}, comparable={"a.md": 3}), STAGE_CHUNKING, "`b.md`"),
        (
            DocumentChunks(stored={"a.md": 3, "b.md": 2}, comparable={"a.md": 3}),
            STAGE_CLAIM_FLOOR,
            "`b.md`",
        ),
        (
            DocumentChunks(stored={"a.md": 3, "b.md": 2}, comparable={"a.md": 3, "b.md": 1}),
            STAGE_CANDIDATES,
            "comparable chunks",
        ),
        # Both sides missing is still one stage and one knob, phrased for two documents.
        (DocumentChunks(stored={}, comparable={}), STAGE_CHUNKING, "`a.md` or `b.md`"),
        # Missing beside a copy the store did NOT keep is an absence like any other.
        (
            DocumentChunks(stored={"a.md": 3}, comparable={"a.md": 3}, copies={"b.md": ("c.md",)}),
            STAGE_CHUNKING,
            "`b.md`",
        ),
    ],
)
def test_the_stage_is_the_first_place_the_pair_could_have_stopped(chunks, stage, named):
    """Ordered by how early the pair was lost: a document not in the store never met a threshold."""
    coverage = {"orderable_document_pairs": 1}
    documents = [("a.md", _2021), ("b.md", _2024)]

    attribution = lost_pair_attribution(coverage, documents, [], chunks)[LOST_PAIR_FIELD]

    assert attribution["documents"] == ["a.md", "b.md"]
    assert attribution["stage"] == stage
    assert attribution["knob"] == STAGE_KNOBS[stage]
    assert named in attribution["reason"]
    assert lost_pair_sentence({LOST_PAIR_FIELD: attribution}).endswith(f"{STAGE_KNOBS[stage]}.")


def test_a_document_the_store_collapsed_into_a_copy_it_kept_is_not_a_chunking_gap():
    """Measured on the shipped demo corpus: exact-duplicate collapse leaves a document chunkless.

    "Rebuild the store" is the wrong advice there -- a rebuild collapses the duplicate again, and
    the claim was never missing: it is in the store under the copy that survived. So this stage is
    the one whose knob is none, which a four-knob list could not express at all.
    """
    chunks = DocumentChunks(
        stored={"a.md": 3, "kept.md": 2},
        comparable={"a.md": 3, "kept.md": 2},
        copies={"b.md": ("kept.md",), "kept.md": ("b.md",)},
    )
    coverage = {"orderable_document_pairs": 1}

    attribution = lost_pair_attribution(coverage, [("a.md", _2021), ("b.md", _2024)], [], chunks)[
        LOST_PAIR_FIELD
    ]

    assert attribution["stage"] == STAGE_DUPLICATE_COLLAPSE
    assert "a copy of `kept.md`" in attribution["reason"]
    assert attribution["knob"] == "none -- read this pair through `kept.md` instead"


# --- end to end: the audit decides the stage ---------------------------------------------------


def _unrelated_corpus(root: Path) -> Path:
    """Two dated documents with nothing in common -- no tier returns the pair they make."""
    return _corpus(root, {"old.md": ("2021-01-01", _body(0)), "new.md": ("2024-01-01", _body(500))})


def test_a_pair_whose_document_never_reached_the_store_names_chunking(tmp_path):
    """The earliest stage: the store the audit read has no chunk of one side at all."""
    corpus = _unrelated_corpus(tmp_path / "corpus")
    result = run_audit(
        corpus,
        AuditParams(effort=TIER_SEMANTIC, cos_threshold=FAKE_COS_THRESHOLD),
        store=_store(corpus, without="new.md"),
    )

    assert result.governance_coverage["orderable_document_pairs"] == 1
    assert _lost(result)["documents"] == ["new.md", "old.md"]
    assert _lost(result)["stage"] == STAGE_CHUNKING
    assert "no chunk of `new.md`" in _lost(result)["reason"]


def test_the_chunking_gap_is_named_even_when_an_unrelated_pair_sorts_first(tmp_path):
    """The whole rule change, decided by the pipeline: three dated documents, two lost pairs.

    `a-old.md` + `b-new.md` sorts first and stops at candidate selection -- both comparable, and
    nothing in common. `z-gap.md` is in the store not at all. Corpus order would name the pair that
    was never going to pair and leave the gap unmentioned.
    """
    corpus = _corpus(
        tmp_path / "corpus",
        {
            "a-old.md": ("2021-01-01", _body(0)),
            "b-new.md": ("2024-01-01", _body(500)),
            "z-gap.md": ("2026-01-01", _body(1000)),
        },
    )
    result = run_audit(
        corpus,
        AuditParams(effort=TIER_SEMANTIC, cos_threshold=FAKE_COS_THRESHOLD),
        store=_store(corpus, without="z-gap.md"),
    )
    documents = [("a-old.md", _2021), ("b-new.md", _2024), ("z-gap.md", _2026)]

    assert result.governance_coverage["orderable_document_pairs"] == 3
    assert not result.findings, "no pair survived, so all three are the scan's to explain"
    assert first_lost_orderable_pair(documents, returned_doc_pairs(result.rows())) == (
        "a-old.md",
        "b-new.md",
    ), "the rule this replaces would have named the unrelated pair"
    assert _lost(result)["stage"] == STAGE_CHUNKING
    assert _lost(result)["documents"] == ["a-old.md", "z-gap.md"]
    assert _lost(result)["knob"] == STAGE_KNOBS[STAGE_CHUNKING]


def test_a_pair_through_a_collapsed_duplicate_names_the_copy_the_store_kept(tmp_path):
    """The shape the shipped demo corpus actually has: two copies, one of them collapsed away.

    The store holds one chunk set for two byte-identical documents, so the pair reaching the
    surviving copy is returned and the pair reaching the collapsed one never can be. Naming
    CHUNKING there would send an operator to rebuild a store that is not wrong.
    """
    corpus = _corpus(
        tmp_path / "corpus",
        {
            "handbook.md": ("2026-01-01", f"{_body(0, BODY_TOKENS - 2)} {_body(900, 2)}"),
            "policy.md": ("2024-01-01", _body(0)),
            "policy_copy.md": ("2024-01-01", _body(0)),
        },
    )
    result = run_audit(
        corpus,
        AuditParams(effort=TIER_SEMANTIC, cos_threshold=FAKE_COS_THRESHOLD),
        store=_store(corpus, without="policy_copy.md"),
    )

    assert result.governance_coverage["orderable_document_pairs"] == 2
    assert _lost(result)["documents"] == ["handbook.md", "policy_copy.md"]
    assert _lost(result)["stage"] == STAGE_DUPLICATE_COLLAPSE
    assert _lost(result)["knob"] == "none -- read this pair through `policy.md` instead"


def test_a_pair_excluded_by_the_claim_token_floor_names_that_floor(tmp_path):
    """In the store and out of the comparison: the knob is the floor, not the candidate list."""
    corpus = _corpus(
        tmp_path / "corpus", {"old.md": ("2021-01-01", _body(0)), "new.md": ("2024-01-01", SHORT)}
    )
    result = run_audit(
        corpus,
        AuditParams(effort=TIER_SEMANTIC, cos_threshold=FAKE_COS_THRESHOLD),
        store=_store(corpus),
    )

    assert result.governance_coverage["orderable_pairs"] == 0
    assert _lost(result)["stage"] == STAGE_CLAIM_FLOOR
    assert "`new.md`" in _lost(result)["reason"]
    assert _lost(result)["knob"] == STAGE_KNOBS[STAGE_CLAIM_FLOOR]


def _budget_corpus(root: Path) -> Path:
    """Fifteen dated documents -- enough cross-document pairs to calibrate a threshold on.

    Fourteen are drawn from disjoint token windows, so every pair among them sits at the same low
    similarity; the last one restates half of its neighbour, so exactly one pair stands above the
    rest. That is what a budget of one candidate pair has to select, leaving the other 104
    orderable document pairs on the floor with nothing else wrong with them.
    """
    documents = {
        f"n{index:02d}.md": (
            f"20{index + 1:02d}-01-01",
            f"{_body(1000 + index * 40)} {_body(0, CORE)}",
        )
        for index in range(14)
    }
    documents["n14.md"] = (
        "2015-01-01",
        f"{_body(1000 + 13 * 40, BODY_TOKENS // 2)} {_body(3000, BODY_TOKENS // 2)} {_body(0, CORE)}",
    )
    return _corpus(root, documents)


def test_a_pair_dropped_by_the_candidate_budget_names_candidate_selection(tmp_path):
    """Comparable on both sides and still not returned: the candidate list is the place left.

    The pair the budget keeps is returned; the first orderable pair it drops is the one attributed,
    and the stage must be the candidate list rather than anything upstream of it.
    """
    corpus = _budget_corpus(tmp_path / "corpus")
    result = run_audit(
        corpus, AuditParams(effort=TIER_SEMANTIC, max_candidate_pairs=1), store=_store(corpus)
    )
    semantic = next(stat for stat in result.tiers if stat.tier == TIER_SEMANTIC)

    assert semantic.extra["cos_threshold_source"] == "calibrated", "the budget set the threshold"
    assert result.governance_coverage["orderable_document_pairs"] == 105
    assert len(result.findings) == 1, "the budget bought exactly one candidate pair"
    assert _lost(result)["documents"] == ["n00.md", "n01.md"]
    assert _lost(result)["stage"] == STAGE_CANDIDATES
    assert _lost(result)["knob"] == STAGE_KNOBS[STAGE_CANDIDATES]
    assert "comparable chunks in the store" in _lost(result)["reason"]


def test_a_run_below_the_semantic_tier_names_the_effort_dial(tmp_path):
    """The four-knob list started with `--effort`, and on a cheap run that is the whole answer."""
    result = run_audit(_unrelated_corpus(tmp_path / "corpus"), AuditParams(effort=TIER_HASH))

    assert _lost(result)["stage"] == STAGE_EFFORT
    assert _lost(result)["knob"] == STAGE_KNOBS[STAGE_EFFORT]


def test_a_run_that_returned_every_orderable_pair_prints_no_attribution(tmp_path):
    """Nothing was lost, so there is no knob -- and a report that named one would be inventing it."""
    body = _body(0)
    corpus = _corpus(
        tmp_path / "corpus", {"old.md": ("2021-01-01", body), "new.md": ("2024-01-01", body)}
    )
    result = run_audit(
        corpus,
        AuditParams(effort=TIER_SEMANTIC, cos_threshold=FAKE_COS_THRESHOLD),
        store=_store(corpus),
    )
    coverage = result.governance_coverage

    assert (coverage["orderable_document_pairs"], coverage["orderable_pairs"]) == (1, 1)
    assert LOST_PAIR_FIELD not in coverage
    assert "did not survive" not in coverage_reading(coverage, zero_delta=True)


# --- the reading: one knob instead of four -----------------------------------------------------


def test_the_retrieval_reading_names_one_knob_when_the_stage_is_attributed(tmp_path):
    """The whole deliverable: the same structural zero, with three of the four knobs removed."""
    corpus = _corpus(
        tmp_path / "corpus", {"old.md": ("2021-01-01", _body(0)), "new.md": ("2024-01-01", SHORT)}
    )
    result = run_audit(
        corpus,
        AuditParams(effort=TIER_SEMANTIC, cos_threshold=FAKE_COS_THRESHOLD),
        store=_store(corpus),
    )
    reading = coverage_reading(result.governance_coverage, zero_delta=True)

    assert "the stage that lost the orderable pair is RETRIEVAL" in reading
    assert RETRIEVAL_KNOBS not in reading, "the four-knob list is what the attribution replaces"
    assert "Earliest stage an orderable document pair was lost at: the CLAIM-TOKEN FLOOR" in reading
    assert reading.endswith(f"One knob: {STAGE_KNOBS[STAGE_CLAIM_FLOOR]}.")


def test_an_audit_that_recorded_no_attribution_still_lists_the_four_knobs():
    """A coverage payload built from rows alone (a recompute) keeps the reading it always had."""
    coverage = {
        "documents": 3,
        "dated_documents": 3,
        "documents_by_field": {"effective_date": 3, "version": 0},
        "document_pairs": 3,
        "orderable_document_pairs": 2,
        "returned_pairs": 1,
        "orderable_pairs": 0,
        "orderable_share": 0.0,
    }
    reading = coverage_reading(coverage, zero_delta=True)

    assert RETRIEVAL_KNOBS in reading
    assert "did not survive" not in reading
