"""Hand-computed pins for character error, word error, and each confusion class.

The strings below are short enough to count by hand. A class count is never added to the
character error: an ordinary deleted letter moves the error rate and leaves every class at zero,
and a ``5`` read as ``S`` is one ``5/S`` whatever the error rate does.
"""

import pytest

from llb.prep.book.metrics.confusion import confusion_counts
from llb.prep.book.metrics.constants import CONFUSION_CLASSES
from llb.prep.book.metrics.page_score import score_page
from llb.prep.book.metrics.reference import RefPage
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.metrics.tables import Cell

# "The cell holds " is 15 characters, ninety-two "n " are 184, and the cell is one "5": 200.
_FIVE_CELL = "The cell holds " + ("n " * 92) + "5"


def _region(y0: int, y1: int) -> dict[str, float]:
    return {"x0": 0.0, "y0": float(y0), "x1": 100.0, "y1": float(y1)}


def _page(blocks: list[dict[str, object]], **extra: object) -> RefPage:
    payload: dict[str, object] = {
        "pdf_page": 1,
        "page_class": "table",
        "printed_page": "57",
        "width": 200,
        "height": 400,
        "blocks": blocks,
    }
    payload.update(extra)
    return RefPage.model_validate(payload)


def _body(text: str, order: int = 0, block_id: str = "b") -> dict[str, object]:
    return {
        "block_id": block_id,
        "kind": "body",
        "region": _region(order * 20, order * 20 + 15),
        "reading_order": order,
        "text": text,
    }


def _submit(
    text: str, *, kind: str = "body", table: tuple[tuple[Cell, ...], ...] | None = None
) -> PageSubmission:
    block = HypBlock("h", kind, None, 0, text, None, table, None, None)
    return PageSubmission("book-ocr/fake", 1, (block,), False)


def test_a_five_read_as_s_in_a_table_cell_is_counted_apart_from_character_error():
    assert len(_FIVE_CELL) == 200
    assert len(_FIVE_CELL.split()) == 96
    reference = _page(
        [
            {
                "block_id": "cell",
                "kind": "table",
                "region": _region(0, 40),
                "reading_order": 0,
                "text": "",
                "table": {"rows": [[_FIVE_CELL]]},
            }
        ]
    )
    changed = _FIVE_CELL[:-1] + "S"
    table = ((Cell(changed),),)
    scored = score_page(reference, _submit("", kind="table", table=table))
    # One character in two hundred. The substitution is the whole of that error, and it is also
    # its own count: 1/200 character error, one 5/S, every other class zero.
    assert scored.character_error_rate.value == pytest.approx(1 / 200)
    assert (
        scored.character_error_rate.value is not None and scored.character_error_rate.value < 0.01
    )
    assert scored.word_error_rate.value == pytest.approx(1 / 96)
    assert scored.missing_words == 1 and scored.duplicated_words == 1
    assert scored.dangerous_substitutions["5/S"] == 1
    assert sum(scored.dangerous_substitutions.values()) == 1
    # One cell relabel on a one-row, one-cell tree: size 3 + 3, distance 1, similarity 5/6.
    assert scored.table_similarity.value == pytest.approx(5 / 6)


def test_an_ordinary_deletion_is_not_a_dangerous_substitution():
    # "alpha beta" is 10 characters. Dropping the final "a" is distance 1, so the rate is 1/10.
    # "beta" and "bet" are different words: one missing, one duplicated. No confusion class fires.
    reference = _page([_body("alpha beta")])
    scored = score_page(reference, _submit("alpha bet"))
    assert scored.character_error_rate.value == pytest.approx(1 / 10)
    assert scored.missing_words == 1 and scored.duplicated_words == 1
    assert all(count == 0 for count in scored.dangerous_substitutions.values())


def test_missing_and_duplicated_words_and_the_word_error():
    # alpha beta gamma (3 words, 16 chars) against alpha gamma gamma.
    # The word alignment substitutes "beta" with "gamma" and the other "gamma" matches,
    # so the word distance is 1 and the rate is 1/3.
    # The multiset still records the lost "beta" and the extra "gamma" as one each.
    # "beta" -> "gamma" is character distance 4, and the shared prefix and suffix add
    # nothing, so the character rate is 4/16.
    reference = _page([_body("alpha beta gamma")])
    scored = score_page(reference, _submit("alpha gamma gamma"))
    assert len("alpha beta gamma") == 16
    assert scored.word_error_rate.value == pytest.approx(1 / 3)
    assert scored.missing_words == 1 and scored.duplicated_words == 1
    assert scored.character_error_rate.value == pytest.approx(4 / 16)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "kind"),
    [
        ("0", "O", "0/O"),
        ("1", "l", "1/l/I"),
        ("1", "I", "1/l/I"),
        ("5", "S", "5/S"),
        ("5", "s", "5/S"),
        ("\u2212", "-", "minus/hyphen/dash"),
        ("-", "\u2013", "minus/hyphen/dash"),
        ("\u2013", "\u2014", "minus/hyphen/dash"),
        ("1.5", "1,5", "decimal/comma"),
        ("1.5", "15", "decimal/comma"),
        ("a", "\u03b1", "latin/greek"),
        ("v", "\u03bd", "latin/greek"),
        ("\u00b0", "o", "unit"),
        ("(", "[", "bracket"),
        ("(3-4)", "(3-5)", "equation-number"),
        ("(3\u22124)", "(3-4)", "minus/hyphen/dash"),
        ("\u22123", "3", "minus/hyphen/dash"),
    ],
)
def test_each_confusion_class_is_one_count_and_the_others_stay_zero(reference, hypothesis, kind):
    counts = confusion_counts(reference, hypothesis, 0)
    assert set(counts) == set(CONFUSION_CLASSES)
    assert counts[kind] == 1
    assert sum(counts.values()) == 1


def test_a_lost_subscript_is_one_script_event():
    counts = confusion_counts("v\u2091", "v", 1)
    assert counts["script"] == 1
    assert sum(counts.values()) == 1
