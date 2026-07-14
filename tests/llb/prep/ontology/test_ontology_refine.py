"""Tests for ontology refine."""

from llb.prep.ontology.constants import PROVENANCE_KIND
from llb.prep.ontology.language import is_ukrainian_dominant
from llb.prep.ontology.models import DocRecord
from llb.prep.ontology.refine import is_circular, refine_drafts
from tests.llb.prep.ontology._ontology_fixtures import DOC1


def test_is_circular_rejects_answer_in_question_or_equal():
    assert is_circular("Що таке столицею?", "столицею", "столицею") is True
    assert is_circular("столицею", "столицею", "столицею") is True
    assert is_circular("Чим є місто для держави?", "столицею", "столицею") is False


def test_ukrainian_output_gate_rejects_foreign_answer_and_allows_latin_proper_name():
    assert is_ukrainian_dominant("Організація Beta є кінцевою сутністю.") is True
    assert is_ukrainian_dominant("More than all the hawks was the brave heart.") is False
    assert (
        is_ukrainian_dominant("Це відповідь: More than all the hawks was the brave heart.") is False
    )


def test_refine_grounds_dedups_and_rejects_circular():
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    drafts = [
        {
            "doc_id": "a.md",
            "question": "Що відомо про Київ?",
            "reference_answer": "України",
            "answer_span": "України",
        },
        {
            "doc_id": "a.md",
            "question": "Що відомо про Київ?",
            "reference_answer": "України",
            "answer_span": "України",
        },  # duplicate question+span -> dropped
        {
            "doc_id": "a.md",
            "question": "Назви Дніпро.",
            "reference_answer": "Дніпро",
            "answer_span": "Дніпро",
        },  # circular (answer in question) -> dropped
        {
            "doc_id": "a.md",
            "question": "Куди тече річка?",
            "reference_answer": "Лондон",
            "answer_span": "Лондон",
        },  # ungrounded -> dropped
    ]
    items = refine_drafts(docs, drafts)
    assert len(items) == 1
    item = items[0]
    assert item.provenance == PROVENANCE_KIND and item.verified is False
    span = item.source_spans[0]
    assert DOC1[span.char_start : span.char_end] == "України"


def test_refine_rejects_non_ukrainian_question_or_reference_answer():
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    drafts = [
        {
            "doc_id": "a.md",
            "question": "Що відомо про Київ?",
            "reference_answer": "The capital of Ukraine.",
            "answer_span": "України",
        },
        {
            "doc_id": "a.md",
            "question": "What is known about Kyiv?",
            "reference_answer": "Столицею є Київ.",
            "answer_span": "Київ",
        },
    ]

    assert refine_drafts(docs, drafts) == []
