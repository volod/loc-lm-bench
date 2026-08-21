"""The gold-item record table and its comparison specification (no Splink, no GPU)."""

import pytest
from llb.goldset.schema import GoldItem, SourceSpan
from llb.prep.ontology.constants import PROVENANCE_KIND, QUESTION_TYPE_MULTI_HOP
from llb.prep.ontology.linkage.constants import (
    QUESTION_TYPE_COLUMN,
    SPAN_BLOCKS_COLUMN,
    SPAN_BLOCK_CHARS,
    SPLIT_COLUMN,
)
from llb.prep.ontology.linkage.records import (
    build_gold_item_spec,
    build_records,
    embed_columns,
    question_type_of,
    span_blocks,
)
from llb.prep.ontology.models import ItemLabels
from tests.llb.prep.ontology.linkage.gold_item_fixtures import HashingBagEmbedder


def _item(item_id: str, question: str, spans: list[tuple[int, int]]) -> GoldItem:
    text = "x" * (max(end for _start, end in spans) + 1)
    return GoldItem(
        id=item_id,
        question=question,
        reference_answer="відповідь",
        source_doc_id="a.md",
        source_spans=[
            SourceSpan(doc_id="a.md", char_start=start, char_end=end, text=text[start:end])
            for start, end in spans
        ],
        provenance=PROVENANCE_KIND,
        split="final",
    )


def test_span_blocks_cover_every_grid_cell_a_span_touches():
    item = _item("a", "Що це?", [(0, 10), (SPAN_BLOCK_CHARS - 1, SPAN_BLOCK_CHARS * 2 + 5)])
    assert span_blocks(item) == ["a.md:0", "a.md:1", "a.md:2"]


def test_two_citations_of_one_sentence_share_a_block_despite_different_offsets():
    left = _item("l", "Питання одне?", [(10, 30)])
    right = _item("r", "Питання два?", [(12, 34)])
    assert set(span_blocks(left)) & set(span_blocks(right))


def test_question_type_prefers_the_recorded_label():
    item = _item("a", "Що таке ліцензія?", [(0, 5)])
    label = ItemLabels(question_type="numeric", difficulty="easy")
    assert question_type_of(item, label) == "numeric"


def test_a_prior_multi_span_item_is_derived_as_multi_hop():
    item = _item("a", "Кому належить компанія, якою керує Alpha?", [(0, 5), (10, 20)])
    assert question_type_of(item) == QUESTION_TYPE_MULTI_HOP


def test_a_prior_single_span_item_is_derived_with_the_drafting_classifier():
    assert question_type_of(_item("a", "Що таке ліцензія?", [(0, 5)])) == "definition"


def test_records_carry_both_roles_and_align_with_their_vectors():
    prior = [_item("p1", "Питання одне?", [(0, 5)])]
    candidates = [_item("c1", "Питання два?", [(0, 5)])]
    embedder = HashingBagEmbedder()
    questions, answers = embed_columns(embedder, [*prior, *candidates])
    records, by_record = build_records(prior, candidates, {}, questions, answers)

    assert [record["role"] for record in records] == ["prior", "candidate"]
    assert by_record[str(records[1]["unique_id"])].id == "c1"
    assert records[0][SPAN_BLOCKS_COLUMN] == ["a.md:0"]
    assert records[1][QUESTION_TYPE_COLUMN]


def test_records_reject_vectors_that_do_not_align():
    prior = [_item("p1", "Питання одне?", [(0, 5)])]
    with pytest.raises(ValueError, match="align"):
        build_records(prior, [], {}, [], [])


def test_the_specification_declares_its_embedding_widths_and_retains_the_split():
    spec = build_gold_item_spec(64, 32)
    widths = {c.column: c.dimension for c in spec.comparisons if c.dimension}
    assert set(widths.values()) == {64, 32}
    # Splits are assigned AFTER deduplication, so agreement on one would price when a field gets
    # filled in rather than whether two items are the same question.
    assert SPLIT_COLUMN in spec.retain_columns
    assert SPLIT_COLUMN not in spec.compared_columns


def test_the_specification_trains_every_comparison_it_blocks_on():
    spec = build_gold_item_spec(8, 8)
    trained = set(spec.compared_columns)
    for rule in spec.em_rules:
        # each pass holds one column fixed; the rules must cover each other
        assert set(rule.expressions) <= trained
    held_fixed = {expression for rule in spec.em_rules for expression in rule.expressions}
    assert all(
        any(expression not in rule.expressions for rule in spec.em_rules)
        for expression in held_fixed
    )


def test_the_shadow_flag_requires_prior_bundles_to_score_against():
    from llb.prep.ontology.pipeline.settings import DraftSettings

    settings = DraftSettings(corpus_root="corpus", dedup_linkage_shadow=True)
    with pytest.raises(ValueError, match="dedup_linkage_shadow requires dedup_against"):
        settings.validate()
