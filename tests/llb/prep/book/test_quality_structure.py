"""Formula fidelity, table trees, boxes, and reading order."""

import pytest

from llb.core.paths import PROJECT_ROOT
from llb.prep.book.metrics.constants import STATUS_ABSENT
from llb.prep.book.metrics.geometry import Box, intersection_over_union, rank_agreement
from llb.prep.book.metrics.latex import (
    canonicalize,
    formula_fidelity,
    latex_parses,
    plain_form,
    terms_of,
)
from llb.prep.book.metrics.page_score import score_page
from llb.prep.book.metrics.reading import reference_reading
from llb.prep.book.metrics.reference import RefPage, load_reference
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.metrics.tables import Cell

_COMPLETE = r"v_{spc}(t_{1}) = v_{E}(t_{1}) + \omega_{E}(t_{1} - t_{0}) = v_{E}(t_{0})"
_DROPPED = r"v_{spc}(t_{1}) = v_{E}(t_{1}) + \omega_{E}(t_{1} - t_{0})"
_FIXTURE = PROJECT_ROOT / "tests/llb/prep/book/fixtures/metrics/reference"


def test_equivalent_latex_spellings_score_equal_and_invalid_latex_is_flagged():
    assert canonicalize(r"v_{E}(t_{0})") == canonicalize(r"v_E(t_0)")
    assert canonicalize(r"v_{{E}}(t_{{0}})") == canonicalize(r"v_{E}(t_{0})")
    assert canonicalize(r"\left(a\right)") == "(a)"
    same = formula_fidelity(r"v_{E}(t_{0})", r"v_{E}(t_{0})", hypothesis_is_latex=True)
    other = formula_fidelity(r"v_{E}(t_{0})", r"v_E(t_0)", hypothesis_is_latex=True)
    assert same.fidelity == other.fidelity == 1.0
    assert latex_parses(r"\frac{a}{b}")
    assert not latex_parses(r"\frac{a}{")
    flagged = score_page(_formula_page(r"\frac{a}{b}"), _formula_submission(r"\frac{a}{"))
    assert flagged.formula_parse_valid is False


def test_a_parsed_formula_with_one_term_removed_scores_below_the_complete_one():
    # Four top-level terms. Dropping the final "= v_{E}(t_{0})" leaves three, so fidelity is 3/4.
    assert len(terms_of(_COMPLETE)) == 4
    assert len(terms_of(_DROPPED)) == 3
    complete = formula_fidelity(_COMPLETE, _COMPLETE, hypothesis_is_latex=True)
    dropped = formula_fidelity(_COMPLETE, _DROPPED, hypothesis_is_latex=True)
    assert complete.parse_valid and dropped.parse_valid
    assert complete.fidelity == 1.0
    assert dropped.fidelity == pytest.approx(3 / 4)
    assert dropped.fidelity is not None and complete.fidelity is not None
    assert dropped.fidelity < complete.fidelity
    page = score_page(_formula_page(_COMPLETE), _formula_submission(_DROPPED))
    assert page.formula_fidelity.value == pytest.approx(3 / 4)
    assert page.formula_parse_valid is True


def test_a_text_only_lane_scores_plain_formula_terms_and_keeps_structure_absent():
    # The four plain readings joined, with the last term (`vE(t0)`) left out. 3/4 of the terms remain.
    terms = [plain_form(term) for term in terms_of(_COMPLETE)]
    assert terms[-1] == "vE(t0)"
    kept = PageSubmission(
        "existing-pdf",
        1,
        (HypBlock("t", "body", None, 0, " ".join(terms[:-1]), None, None, None, None),),
        True,
    )
    scored = score_page(_formula_page(_COMPLETE), kept)
    assert scored.formula_fidelity.value == pytest.approx(3 / 4)
    assert scored.formula_parse_valid is None
    page = load_reference(_FIXTURE)[0]
    reversed_text = " ".join(
        reference_reading(block)
        for block in sorted(page.blocks, key=lambda item: item.reading_order, reverse=True)
    )
    backwards = PageSubmission(
        "existing-pdf",
        1,
        (HypBlock("t", "body", None, 0, reversed_text, None, None, None, None),),
        True,
    )
    order = score_page(page, backwards)
    assert order.reading_order_agreement.value == 0.0
    assert order.table_similarity.status == STATUS_ABSENT
    assert order.figure_overlap.status == STATUS_ABSENT


def test_tree_edit_table_similarity_box_overlap_and_reading_order():
    # 2x2 tree: 1 table + 2 rows + 4 cells = 7. One cell relabel costs 1. 1 - 1/14 = 13/14.
    reference = _table_page([["a", "b"], ["c", "d"]])
    changed = _table_submission([["a", "b"], ["c", "e"]])
    assert score_page(reference, changed).table_similarity.value == pytest.approx(13 / 14)
    assert (
        score_page(reference, _table_submission([["a", "b"], ["c", "d"]])).table_similarity.value
        == 1.0
    )
    # (0,0)-(10,10) against (0,0)-(10,5): intersection 50, union 100, overlap 1/2.
    wide, short = Box(0, 0, 10, 10), Box(0, 0, 10, 5)
    assert intersection_over_union(wide, short) == pytest.approx(1 / 2)
    overlap = score_page(_figure_page(10), _figure_submission(5))
    assert overlap.figure_overlap.value == pytest.approx(1 / 2)
    # Orders (0, 1, 2) against (0, 2, 1): the pair (1, 2) disagrees and the other two agree.
    assert rank_agreement([(0, 0), (1, 2), (2, 1)]) == pytest.approx(2 / 3)
    ordered = score_page(_columns(), _columns(order=(0, 2, 1)))
    assert ordered.reading_order_agreement.value == pytest.approx(2 / 3)


def test_caption_association_and_printed_page_are_exact():
    linked = score_page(_figure_page(10), _figure_submission(10))
    assert linked.caption_association.value == 1.0
    assert linked.printed_page_accuracy.value == 1.0
    wrong = _figure_submission(10, caption_id="other", printed="58")
    missed = score_page(_figure_page(10), wrong)
    assert missed.caption_association.value == 0.0
    assert missed.printed_page_accuracy.value == 0.0


def _formula_page(latex: str) -> RefPage:
    return RefPage.model_validate(
        {
            "pdf_page": 1,
            "page_class": "formula",
            "printed_page": "57",
            "width": 200,
            "height": 80,
            "blocks": [
                {
                    "block_id": "f",
                    "kind": "formula",
                    "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
                    "reading_order": 0,
                    "text": latex,
                    "latex": latex,
                }
            ],
        }
    )


def _formula_submission(latex: str) -> PageSubmission:
    block = HypBlock("f", "formula", Box(0, 0, 100, 20), 0, latex, latex, None, None, None)
    return PageSubmission("book-ocr/fake", 1, (block,), False)


def _table_page(rows: list[list[str]]) -> RefPage:
    return RefPage.model_validate(
        {
            "pdf_page": 1,
            "page_class": "table",
            "printed_page": "57",
            "width": 200,
            "height": 80,
            "blocks": [
                {
                    "block_id": "t",
                    "kind": "table",
                    "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 40},
                    "reading_order": 0,
                    "text": "",
                    "table": {"rows": rows},
                }
            ],
        }
    )


def _table_submission(rows: list[list[str]]) -> PageSubmission:
    table = tuple(tuple(Cell(text) for text in row) for row in rows)
    block = HypBlock("t", "table", Box(0, 0, 100, 40), 0, "", None, table, None, None)
    return PageSubmission("book-ocr/fake", 1, (block,), False)


def _figure_page(height: float) -> RefPage:
    return RefPage.model_validate(
        {
            "pdf_page": 1,
            "page_class": "figure",
            "printed_page": "57",
            "width": 200,
            "height": 80,
            "blocks": [
                {
                    "block_id": "fig",
                    "kind": "figure",
                    "region": {"x0": 0, "y0": 0, "x1": 10, "y1": height},
                    "reading_order": 0,
                    "text": "",
                    "caption_id": "cap",
                },
                {
                    "block_id": "cap",
                    "kind": "caption",
                    "region": {"x0": 0, "y0": 40, "x1": 40, "y1": 55},
                    "reading_order": 1,
                    "text": "Fig. 1.",
                },
                {
                    "block_id": "num",
                    "kind": "page-number",
                    "region": {"x0": 0, "y0": 60, "x1": 20, "y1": 75},
                    "reading_order": 2,
                    "text": "57",
                },
            ],
        }
    )


def _figure_submission(
    height: float, caption_id: str = "cap", printed: str = "57"
) -> PageSubmission:
    blocks = (
        HypBlock("fig", "figure", Box(0, 0, 10, height), 0, "", None, None, caption_id, None),
        HypBlock("cap", "caption", Box(0, 40, 40, 55), 1, "Fig. 1.", None, None, None, None),
        HypBlock("other", "caption", Box(150, 40, 190, 55), 3, "other", None, None, None, None),
        HypBlock("num", "page-number", Box(0, 60, 20, 75), 2, printed, None, None, None, None),
    )
    return PageSubmission("book-ocr/fake", 1, blocks, False)


def _columns(order: tuple[int, int, int] = (0, 1, 2)) -> PageSubmission | RefPage:
    texts = ("left column", "middle column", "right column")
    blocks = []
    for index, text in enumerate(texts):
        blocks.append(
            {
                "block_id": f"c{index}",
                "kind": "body",
                "region": {"x0": 0, "y0": index * 20, "x1": 80, "y1": index * 20 + 15},
                "reading_order": order[index],
                "text": text,
            }
        )
    if order == (0, 1, 2):
        return RefPage.model_validate(
            {
                "pdf_page": 1,
                "page_class": "normal",
                "printed_page": "57",
                "width": 200,
                "height": 80,
                "blocks": blocks,
            }
        )
    hyp = tuple(
        HypBlock(
            f"c{index}",
            "body",
            Box(0, index * 20, 80, index * 20 + 15),
            order[index],
            texts[index],
            None,
            None,
            None,
            None,
        )
        for index in range(3)
    )
    return PageSubmission("book-ocr/fake", 1, hyp, False)
