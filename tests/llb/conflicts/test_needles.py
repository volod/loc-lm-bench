"""The needle lane: which gold answers can be found in more than one document.

This is the corroborating signal for the tree's findings. It is derived from the gold set rather
than from corpus geometry, so when the two agree that is evidence rather than a restatement of the
same measurement.
"""

from llb.conflicts.needles import analyze_needles
from llb.goldset.schema import GoldItem, SourceSpan

from conflict_helpers import (
    DOC_2024,
    DOC_ARCHIVE,
    FAKE_COS_THRESHOLD,
    FIXTURE_CORPUS,
    fake_store_view,
)


def gold_item(item_id: str, doc_id: str, needle: str) -> GoldItem:
    """A gold item whose span is the literal `needle` inside `doc_id`."""
    text = (FIXTURE_CORPUS / doc_id).read_text(encoding="utf-8")
    start = text.index(needle)
    return GoldItem(
        id=item_id,
        question=f"питання про {item_id}",
        reference_answer="відповідь",
        source_doc_id=doc_id,
        source_spans=[
            SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(needle), text=needle)
        ],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def analyze(items):
    store = fake_store_view()
    return analyze_needles(items, store.chunks, store.vectors, cos_threshold=FAKE_COS_THRESHOLD)


def test_answer_duplicated_in_another_document_is_flagged_ambiguous():
    """The e-appeals text lives in both the standalone note and the 2024 article."""
    rows, report = analyze([gold_item("dup", DOC_2024, "Електронне звернення подається")])
    assert report["ambiguous_items"] == 1
    assert rows[0].is_ambiguous
    assert "e-appeals-note.md" in rows[0].foreign_docs


def test_answer_unique_to_one_document_is_not_flagged():
    rows, report = analyze([gold_item("uniq", DOC_ARCHIVE, "Температурний режим у сховищі")])
    assert report["ambiguous_items"] == 0
    assert rows[0].foreign_docs == []


def test_fraction_reports_over_all_items():
    items = [
        gold_item("dup", DOC_2024, "Електронне звернення подається"),
        gold_item("uniq", DOC_ARCHIVE, "Температурний режим у сховищі"),
    ]
    _, report = analyze(items)
    assert report["items"] == 2
    assert report["non_unique_needle_fraction"] == 0.5


def test_span_matching_no_chunk_is_reported_separately_not_as_unique():
    """An unlocatable gold span is missing evidence, which is not the same as unambiguous."""
    item = gold_item("dup", DOC_2024, "Електронне звернення подається")
    orphan = item.model_copy(
        update={
            "id": "orphan",
            "source_spans": [
                SourceSpan(doc_id="does-not-exist.md", char_start=0, char_end=5, text="абвгд")
            ],
        }
    )
    rows, report = analyze([orphan])
    assert report["unlocated_items"] == 1
    assert report["unlocated_ids"] == ["orphan"]
    assert rows[0].gold_chunks == 0


def test_empty_goldset_reports_zero_rather_than_dividing_by_zero():
    _, report = analyze([])
    assert report["items"] == 0
    assert report["non_unique_needle_fraction"] == 0.0
