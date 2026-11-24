"""Cases where an absent or changed technical detail can give a pilot spurious credit."""

from dataclasses import replace

import pytest

from llb.prep.book.metrics.confusion import confusion_counts
from llb.prep.book.metrics.geometry import Box
from llb.prep.book.metrics.latex import formula_fidelity
from llb.prep.book.metrics.page_score import score_page
from llb.prep.book.metrics.reference import RefPage
from llb.prep.book.metrics.submission import HypBlock, PageSubmission


def _page(**fields: object) -> RefPage:
    return RefPage.model_validate(
        dict(pdf_page=1, page_class="normal", width=100, height=100, blocks=[], printed_page=None)
        | fields
    )


def _submission(*blocks: HypBlock) -> PageSubmission:
    return PageSubmission("book-ocr/fake", 1, blocks, False)


def test_unnumbered_blank_page_and_number_in_body() -> None:
    body = HypBlock("t", "body", None, 0, "see page 12", None, None, None, None)
    assert score_page(_page(), _submission()).printed_page_accuracy.value == 1
    missing = replace(_submission(), present=False)
    assert score_page(_page(), missing).printed_page_accuracy.value == 0
    assert (
        score_page(
            _page(), _submission(replace(body, kind="page-number"))
        ).printed_page_accuracy.value
        == 0
    )
    assert score_page(_page(printed_page="12"), _submission(body)).printed_page_accuracy.value == 0
    assert (
        score_page(_page(), replace(_submission(body), text_only=True)).printed_page_accuracy.status
        == "absent"
    )


def test_uncaptioned_figure_scores_only_with_matching_unlinked_box() -> None:
    page = _page(
        blocks=[
            dict(
                block_id="f", kind="figure", reading_order=0, region=dict(x0=0, y0=0, x1=50, y1=50)
            )
        ]
    )
    figure = HypBlock("f", "figure", Box(0, 0, 50, 50), 0, "", None, None, None, None)
    assert score_page(page, _submission(figure)).caption_association.value == 1
    assert score_page(page, _submission()).caption_association.value == 0
    assert (
        score_page(page, _submission(replace(figure, caption_id="c"))).caption_association.value
        == 0
    )


@pytest.mark.parametrize("is_latex", [False, True])
def test_changed_sign_and_dropped_repeat_lower_fidelity(is_latex: bool) -> None:
    assert confusion_counts("x=a-b", "x=a+b", 0)["minus/hyphen/dash"] == 1
    assert formula_fidelity(
        "x=a+b", "x=a-b", hypothesis_is_latex=is_latex
    ).fidelity == pytest.approx(2 / 3)
    assert formula_fidelity(
        "x=a-b", "x=a+b", hypothesis_is_latex=is_latex
    ).fidelity == pytest.approx(2 / 3)
    assert formula_fidelity("x=a+a", "x=a", hypothesis_is_latex=is_latex).fidelity == pytest.approx(
        2 / 3
    )


def test_extra_malformed_formula_on_text_page_fails_parse_gate() -> None:
    extra = HypBlock("f", "formula", None, 0, "a+(", None, None, None, None)
    assert score_page(_page(), _submission(extra)).formula_parse_valid is False
