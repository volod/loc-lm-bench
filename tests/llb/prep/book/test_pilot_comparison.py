"""The pilot decision: margins, the dangerous-substitution veto, and the pre-draft."""

import json
from itertools import product

import pytest

from llb.prep.book.compare.assemble import compare_submissions
from llb.prep.book.compare.bootstrap import interval_status, page_bootstrap_interval
from llb.prep.book.compare.constants import BASELINE_ENGINE, REOPEN_WHEN, ROLE_VARIANT
from llb.prep.book.compare.inputs import reference_digest
from llb.prep.book.metrics.constants import LANE_EXISTING
from tests.llb.prep.book.compare_factory import (
    BASELINE,
    accepts_for,
    candidate,
    page,
    serves,
    submission,
)


def test_page_bootstrap_interval_is_the_resample_range() -> None:
    values = [0.2, -0.1, 0.5]
    resampled = [sum(draw) / len(values) for draw in product(values, repeat=len(values))]
    low, high = page_bootstrap_interval(values)
    assert (low, high) == (float(min(values)), float(max(values)))
    assert min(resampled) == pytest.approx(low)
    assert max(resampled) == pytest.approx(high)
    assert interval_status(0.0, 1.0) == "inconclusive"
    assert interval_status(0.5, 0.5) == "gain"
    assert interval_status(-0.2, 0.4) == "inconclusive"
    with pytest.raises(ValueError):
        page_bootstrap_interval(())


def test_a_gain_on_every_page_selects_the_candidate_and_records_who_won_nothing() -> None:
    pages = [page("normal", 1, "abcd"), page("normal", 2, "abcd")]
    report, _config = _compare(
        pages,
        [
            candidate("book-ocr/reader", "reader"),
            candidate("book-ocr/worse", "reader", vram_mb=50),
        ],
        [
            *_baseline(pages, ["abxx", "abxx"]),
            submission("book-ocr/reader", 1, "abcd"),
            submission("book-ocr/reader", 2, "abcd"),
            submission("book-ocr/worse", 1, "xxxx"),
            submission("book-ocr/worse", 2, "xxxx"),
        ],
    )
    choice = report.classes[0]
    assert choice.pages == 2
    assert choice.winner_lane == "book-ocr/reader"
    assert choice.configured_engine == "reader"
    winner = _lane(choice, "book-ocr/reader")
    assert winner.margin is not None
    assert winner.margin.status == "gain"
    assert winner.margin.low > 0
    assert winner.margin.point == pytest.approx(0.5)
    assert winner.blocked_dangerous is False
    assert "book-ocr/worse" in report.won_no_class
    assert choice.winner_lane not in report.won_no_class
    dumped = report.model_dump(mode="json")
    assert "score" not in _keys(dumped)
    assert sum(winner.metrics.dangerous_substitutions.values()) == 0


def test_a_straddle_keeps_the_existing_lane() -> None:
    pages = [page("normal", 1, "abcd"), page("normal", 2, "abcd")]
    report, _config = _compare(
        pages,
        [candidate("book-ocr/reader", "reader")],
        [
            *_baseline(pages, ["abxx", "abcd"]),
            submission("book-ocr/reader", 1, "abcd"),
            submission("book-ocr/reader", 2, "abxx"),
        ],
    )
    choice = report.classes[0]
    assert choice.winner_lane == LANE_EXISTING
    assert choice.configured_engine == BASELINE_ENGINE
    assert "inconclusive" in choice.rationale


def test_a_higher_dangerous_rate_blocks_a_character_gain() -> None:
    pages = [page("normal", 1, "aa5"), page("normal", 2, "aa5")]
    report, _config = _compare(
        pages,
        [candidate("book-ocr/reader", "reader")],
        [
            *_baseline(pages, ["xx5", "xx5"]),
            submission("book-ocr/reader", 1, "aaS"),
            submission("book-ocr/reader", 2, "aaS"),
        ],
    )
    choice = report.classes[0]
    row = _lane(choice, "book-ocr/reader")
    assert row.margin is not None
    assert row.margin.point == pytest.approx(1 / 3)
    assert row.blocked_dangerous is True
    assert choice.winner_lane == LANE_EXISTING
    example = _example(report, "5/S")
    assert example.found is True
    assert "5" in (example.reference or "")
    assert "S" in (example.hypothesis or "")


def test_an_inconclusive_dangerous_interval_does_not_block_an_equal_pooled_rate() -> None:
    pages = [page("normal", 1, "aa5"), page("normal", 2, "aa5")]
    report, _config = _compare(
        pages,
        [candidate("book-ocr/reader", "reader")],
        [
            *_baseline(pages, ["xxS", "xx5"]),
            submission("book-ocr/reader", 1, "aa5"),
            submission("book-ocr/reader", 2, "aaS"),
        ],
    )
    choice = report.classes[0]
    row = _lane(choice, "book-ocr/reader")
    assert row.blocked_dangerous is False
    assert row.dangerous_margin is not None
    assert row.dangerous_margin.status == "inconclusive"
    assert row.dangerous_margin.low == pytest.approx(-1 / 3)
    assert row.dangerous_margin.high == pytest.approx(1 / 3)
    assert choice.winner_lane == "book-ocr/reader"


def test_pre_draft_agreement_does_not_choose_the_winner() -> None:
    pages = [page("normal", 1, "abcdef")]
    draft = [page("normal", 1, "abcdefZZ")]
    report, _config = _compare(
        pages,
        [candidate("book-ocr/matched", "reader"), candidate("book-ocr/anchored", "reader")],
        [
            *_baseline(pages, ["abcdef"]),
            submission("book-ocr/matched", 1, "abcdef"),
            submission("book-ocr/anchored", 1, "abcdefZZ"),
        ],
        predraft=draft,
    )
    choice = report.classes[0]
    assert choice.winner_lane == LANE_EXISTING
    matched = _lane(choice, "book-ocr/matched")
    anchored = _lane(choice, "book-ocr/anchored")
    assert matched.predraft_character_agreement == pytest.approx(0.75)
    assert anchored.predraft_character_agreement == pytest.approx(1.0)
    assert matched.metrics.character_error_rate.value == pytest.approx(0.0)
    assert anchored.metrics.character_error_rate.value == pytest.approx(1 / 3)


def test_no_gain_records_a_negative_and_keeps_docling() -> None:
    pages = [page("normal", 1, "abcd"), page("normal", 2, "abcd")]
    report, config = _compare(
        pages,
        [candidate("book-ocr/reader", "reader")],
        [
            *_baseline(pages, ["abcd", "abcd"]),
            submission("book-ocr/reader", 1, "xxxx"),
            submission("book-ocr/reader", 2, "xxxx"),
        ],
    )
    assert report.negative is True
    assert report.reopen_when == REOPEN_WHEN
    assert report.classes[0].configured_engine == BASELINE_ENGINE
    ocr = _stage(config, "ocr")
    assert ocr["params"]["engine"] == BASELINE_ENGINE
    assert ocr["params"]["ocr_language"] == "eng"


def test_equation_numbers_and_digit_substitutions_are_separate_examples() -> None:
    pages = [page("dense_formulas", 1, "5 (3-4)")]
    report, _config = _compare(
        pages,
        [candidate("book-ocr/reader", "reader")],
        [*_baseline(pages, ["5 (3-4)"]), submission("book-ocr/reader", 1, "S (3-5)")],
    )
    equation = _example(report, "equation-number")
    digits = _example(report, "5/S")
    assert equation.found and equation.reference == "(3-4)" and equation.hypothesis == "(3-5)"
    assert digits.found is True
    assert "5" in (digits.reference or "") and "S" in (digits.hypothesis or "")


def test_reference_digest_ignores_input_order() -> None:
    pages = [page("normal", 2, "b"), page("normal", 1, "a")]
    assert reference_digest(pages) == reference_digest(list(reversed(pages)))


def test_variant_policy_is_adopted_per_class_and_one_dewarp_method_wins() -> None:
    curved = [page("severe_curvature", 1, "a" * 10), page("severe_curvature", 2, "a" * 10)]
    gutter = [page("gutter_corner", 3, "b" * 10), page("gutter_corner", 4, "b" * 10)]
    normal = [page("normal", 5, "flat text."), page("normal", 6, "flat text.")]
    pages = [*curved, *gutter, *normal]
    original = "book-ocr/rapidocr"
    uvdoc = "book-ocr/rapidocr@dewarp-uvdoc"
    opencv = "book-ocr/rapidocr@dewarp-opencv"
    deskew = "book-ocr/rapidocr@deskew"
    report, config = _compare(
        pages,
        [
            candidate(original, "rapidocr", role=ROLE_VARIANT, vram_mb=0),
            candidate(uvdoc, "rapidocr", role=ROLE_VARIANT, variant="dewarp", dewarp="uvdoc"),
            candidate(opencv, "rapidocr", role=ROLE_VARIANT, variant="dewarp", dewarp="opencv"),
            candidate(deskew, "rapidocr", role=ROLE_VARIANT, variant="deskew"),
        ],
        [
            *_baseline(pages, [item.blocks[0].text for item in pages]),
            *_rapid(original, curved, "x" * 10),
            *_rapid(uvdoc, curved, "a" * 10),
            *_rapid(opencv, curved, "x" * 10),
            *_rapid(deskew, curved, "x" * 10),
            *_rapid(original, gutter, "y" * 5 + "b" * 5),
            *_rapid(uvdoc, gutter, "y" * 5 + "b" * 5),
            *_rapid(opencv, gutter, "b" * 10),
            *_rapid(deskew, gutter, "y" * 5 + "b" * 5),
            *_rapid(original, normal, "flat text."),
            *_rapid(uvdoc, normal, "flat xxxx."),
            *_rapid(opencv, normal, "flat xxxx."),
            *_rapid(deskew, normal, "flat text."),
        ],
    )
    by_class = {choice.page_class: choice for choice in report.classes}
    assert by_class["severe_curvature"].variant == "dewarp"
    assert by_class["severe_curvature"].dewarp == "uvdoc"
    assert by_class["gutter_corner"].variant == "original"
    assert "dewarp method conflict" in by_class["gutter_corner"].rationale
    assert by_class["normal"].variant == "original"
    assert all(choice.winner_lane == LANE_EXISTING for choice in report.classes)
    ocr = _stage(config, "ocr")
    assert ocr["params"]["variant"] == "original"
    assert ocr["params"]["variant_by_page_class"] == {"severe_curvature": "dewarp"}
    variants = _stage(config, "variants")
    assert variants["params"]["dewarp"] == "uvdoc"
    assert deskew in report.variants_not_adopted


def _compare(pages, lanes, submissions, *, predraft=None, **kwargs):
    return compare_submissions(
        pages,
        pages if predraft is None else predraft,
        submissions,
        lanes,
        language="eng",
        accepts=accepts_for(pages),
        baseline_vram_mb=2970,
        class_page_counts={},
        serves=serves,
        **kwargs,
    )


def _baseline(pages, texts):
    return [
        submission(BASELINE, item.pdf_page, text, text_only=True)
        for item, text in zip(pages, texts)
    ]


def _rapid(lane: str, pages, text: str):
    return [submission(lane, item.pdf_page, text) for item in pages]


def _lane(choice, lane: str):
    return next(row for row in choice.lanes if row.lane == lane)


def _example(report, kind: str):
    return next(row for row in report.examples if row.confusion_class == kind)


def _stage(config, name: str):
    return next(
        stage for stage in config["stages"] if isinstance(stage, dict) and stage["name"] == name
    )


def _keys(node) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        found.update(node)
        for value in node.values():
            found.update(_keys(value))
    elif isinstance(node, list):
        for item in node:
            found.update(_keys(item))
    return found


def test_report_json_has_no_folded_score() -> None:
    pages = [page("normal", 1, "abcd")]
    report, _config = _compare(
        pages,
        [],
        [*_baseline(pages, ["abcd"])],
    )
    payload = json.dumps(report.model_dump(mode="json"))
    assert '"score"' not in payload
