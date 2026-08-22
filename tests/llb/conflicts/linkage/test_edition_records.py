"""The record table, its blocking column, and the specification built over the run's own cutoffs.

These run in the base install: nothing here needs Splink, and the property that matters most --
that the exploding blocking column generates exactly the lexical tier's candidate list -- is a
statement about two Python functions, not about the engine.
"""

import pytest

from llb.conflicts.corpus import load_corpus_docs
from llb.conflicts.constants import MAX_SHINGLE_DOC_FREQUENCY
from llb.conflicts.linkage.constants import (
    BLOCK_KEY_COLUMN,
    BLOCK_SHINGLES_COLUMN,
    DOC_ID_COLUMN,
    EFFECTIVE_DATE_COLUMN,
    SHINGLES_COLUMN,
    SOURCE_SYSTEM_COLUMN,
    TITLE_COLUMN,
)
from llb.conflicts.linkage.records import (
    build_edition_spec,
    build_records,
    discriminative,
    document_title,
)
from llb.conflicts.tiers.lexical import candidate_pairs, shingles
from llb.linkage.constants import KIND_SET_OVERLAP
from llb.linkage.records import column_types

from tests.llb.conflicts.linkage.conftest import EDITIONS_CORPUS


def _shingle_sets(docs):
    return [shingles(doc.body) for doc in docs]


def test_discriminative_blocking_reproduces_the_tiers_candidate_list(edition_docs):
    """Exploding on the blocking column must generate the pairs `candidate_pairs` returns.

    That equality is the whole reason the column exists: the fit prices the lexical tier's OWN
    candidates, so a difference between the two rankings cannot be a difference in what was looked
    at.
    """
    sets = _shingle_sets(edition_docs)
    blocking = discriminative(sets, MAX_SHINGLE_DOC_FREQUENCY)
    exploded = {
        (left, right)
        for left in range(len(blocking))
        for right in range(left + 1, len(blocking))
        if blocking[left] & blocking[right]
    }
    assert exploded == candidate_pairs(sets)


def test_the_blocking_column_is_a_subset_of_the_scored_column(edition_docs):
    """The overlap measures read the WHOLE set; only the blocking reads the discriminative part."""
    sets = _shingle_sets(edition_docs)
    assert all(
        block <= whole
        for block, whole in zip(discriminative(sets, MAX_SHINGLE_DOC_FREQUENCY), sets)
    )


def test_a_shingle_every_document_carries_is_dropped_from_the_blocking_column_only():
    """Boilerplate pairs every document with every other while saying nothing about identity.

    The planted corpus is too small for any of its shingles to reach the frequency limit, so the
    drop is asserted on a set list built to cross it -- the tier's own rule, at its own scale.
    """
    common, unique = 1, 2
    sets = [{common, unique + index} for index in range(10)]
    blocking = discriminative(sets, 0.5)
    assert all(common not in block for block in blocking)
    assert all(unique + index in blocking[index] for index in range(10))
    assert all(common in whole for whole in sets)


def test_title_is_case_folded_so_a_reissue_agrees_with_its_original(edition_docs):
    """The normalized-duplicate edition differs from its original only in case and punctuation."""
    by_id = {doc.doc_id: doc for doc in edition_docs}
    original = document_title(by_id["appeals/polozhennia-2022.md"])
    shouted = document_title(by_id["appeals/polozhennia-2022-sharepoint.md"])
    assert original and original == shouted


def test_a_record_carries_every_column_the_specification_declares(edition_docs):
    docs = list(edition_docs)
    records = build_records(docs, _shingle_sets(docs), MAX_SHINGLE_DOC_FREQUENCY)
    spec = build_edition_spec(
        jaccard_threshold=0.8,
        containment_threshold=0.9,
        match_threshold=0.9,
        random_match_probability=0.01,
    )
    assert len(records) == len(docs)
    for record in records:
        assert set(column_types(spec)) <= set(record)
    assert records[0][DOC_ID_COLUMN] == docs[0].doc_id
    assert column_types(spec)[BLOCK_SHINGLES_COLUMN] == "VARCHAR[]"
    assert column_types(spec)[SHINGLES_COLUMN] == "VARCHAR[]"
    assert column_types(spec)[BLOCK_KEY_COLUMN] == "VARCHAR"


def test_the_ladders_are_built_at_this_runs_own_cutoffs():
    """The top rung of each overlap ladder IS the cutoff the lexical tier decides on."""
    spec = build_edition_spec(
        jaccard_threshold=0.75,
        containment_threshold=0.85,
        match_threshold=0.9,
        random_match_probability=0.01,
    )
    overlap = next(c for c in spec.comparisons if c.kind == KIND_SET_OVERLAP)
    assert overlap.thresholds[0] == 0.75
    assert overlap.containment_thresholds[0] == 0.85
    assert overlap.thresholds[1] < 0.75 and overlap.containment_thresholds[1] < 0.85


def test_a_cutoff_at_the_ladder_step_does_not_produce_a_second_rung_below_zero():
    spec = build_edition_spec(
        jaccard_threshold=0.2,
        containment_threshold=0.3,
        match_threshold=0.9,
        random_match_probability=0.01,
    )
    overlap = next(c for c in spec.comparisons if c.kind == KIND_SET_OVERLAP)
    assert overlap.thresholds == (0.2,)
    assert overlap.containment_thresholds == (0.3,)


def test_the_specification_compares_the_four_fields_and_blocks_by_exploding_shingles():
    spec = build_edition_spec(
        jaccard_threshold=0.8,
        containment_threshold=0.9,
        match_threshold=0.9,
        random_match_probability=0.01,
    )
    assert spec.compared_columns == (
        SHINGLES_COLUMN,
        TITLE_COLUMN,
        SOURCE_SYSTEM_COLUMN,
        EFFECTIVE_DATE_COLUMN,
    )
    assert spec.exploded_columns == (BLOCK_SHINGLES_COLUMN,)
    assert all(not rule.explodes for rule in spec.training_rules)
    assert spec.min_level_probability > 0
    assert spec.retain_matching_columns is False


def test_an_empty_corpus_document_still_produces_a_record(tmp_path):
    """A document with no heading and no governance is a record with empty text columns.

    It must not be dropped: a corpus position the record table skips is a document the pair count
    silently stops being about.
    """
    (tmp_path / "bare.md").write_text("одне слово тут\n", encoding="utf-8")
    docs = load_corpus_docs(tmp_path)
    records = build_records(docs, _shingle_sets(docs), MAX_SHINGLE_DOC_FREQUENCY)
    assert len(records) == 1
    assert records[0][TITLE_COLUMN] == ""
    assert records[0][EFFECTIVE_DATE_COLUMN] == ""


@pytest.mark.parametrize("corpus", [EDITIONS_CORPUS])
def test_the_planted_corpus_is_above_the_lane_floor(corpus):
    """The fixture exists to be fittable; a plant below the floor would only test the decline."""
    from llb.conflicts.linkage.constants import MIN_LINKAGE_DOCUMENTS

    assert len(load_corpus_docs(corpus)) >= MIN_LINKAGE_DOCUMENTS
