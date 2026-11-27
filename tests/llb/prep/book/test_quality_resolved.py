"""The resolved pipeline lane beside the raw engines and the existing PDF lane."""

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from llb.cli import app
from llb.prep.book.metrics.constants import (
    FIELD_CAPTION,
    FIELD_FIGURE,
    FIELD_FORMULA,
    FIELD_TABLE,
    LANE_RESOLVED,
    STATUS_ABSENT,
    STATUS_SCORED,
)
from llb.prep.book.metrics.geometry import nearest_links
from llb.prep.book.metrics.report import ClassScore, ScoreReport
from llb.prep.book.metrics.resolved import load_resolved
from llb.prep.book.metrics.score import score_reference
from tests.llb.prep.book.resolved_score_cases import (
    caption_case,
    competing_case,
    formula_case,
    hand_case,
    unmapped_case,
    write_adopted,
)

_RUNNER = CliRunner()


def test_class_totals_match_the_hand_count(tmp_path: Path) -> None:
    run, reference = hand_case(tmp_path)
    report = score_reference(reference, run)
    resolved = _lane(report, LANE_RESOLVED)
    assert [page.pdf_page for page in resolved.pages] == [1, 2, 3]
    assert resolved.missing_submissions == 1
    _assert_prose(resolved.by_page_class["prose"])
    _assert_glyph(resolved.by_page_class["glyph"])
    page_class = {page.pdf_page: page.page_class for page in resolved.pages}
    assert page_class == {1: "prose", 2: "prose", 3: "glyph"}


def test_a_variant_switch_and_an_overlay_edit_touch_one_page(tmp_path: Path) -> None:
    run, reference = hand_case(tmp_path)
    before = _pages(score_reference(reference, run))
    write_adopted(run, 1, "dog", "deskew")
    switched = _pages(score_reference(reference, run))
    assert _changed(before, switched) == {(LANE_RESOLVED, 1)}
    assert switched[(LANE_RESOLVED, 1)]["character_error_rate"]["value"] == 1.0
    assert before[("book-ocr/alpha", 1)]["character_error_rate"]["value"] == 0.0
    write_adopted(run, 1, "cat", "original")
    restored = _pages(score_reference(reference, run))
    write_adopted(run, 3, "5", "original")
    edited = _pages(score_reference(reference, run))
    assert _changed(restored, edited) == {(LANE_RESOLVED, 3)}
    assert edited[(LANE_RESOLVED, 3)]["character_error_rate"]["value"] == 0.0
    assert edited[(LANE_RESOLVED, 3)]["dangerous_substitutions"]["5/S"] == 0
    assert edited[("book-ocr/alpha", 3)]["dangerous_substitutions"]["5/S"] == 1


def test_the_printed_caption_is_credited_and_a_wrong_folio_is_not(tmp_path: Path) -> None:
    run, reference = caption_case(tmp_path)
    report = score_reference(reference, run)
    resolved = _page(report, LANE_RESOLVED, 1)
    raw = _page(report, "book-ocr/alpha", 1)
    submission = next(item for item in load_resolved(run) if item.pdf_page == 1)
    text = " ".join(block.text for block in submission.blocks)
    assert re.search(r"(?<!\w)57(?!\w)", text)
    # The vault prints the caption whose figure number the page prints, not the reading-order
    # neighbour, so the score credits that association.
    linked = next(block for block in submission.blocks if block.block_id == "p1-b1")
    assert linked.caption_id == "p1-b3"
    assert resolved.caption_association.value == 1.0
    assert resolved.printed_page_accuracy.value == 0.0
    assert raw.printed_page_accuracy.value == 1.0


def test_competing_captions_earn_nothing_when_a_nearest_guess_would_pass(tmp_path: Path) -> None:
    run, reference = competing_case(tmp_path)
    report = score_reference(reference, run)
    resolved = _page(report, LANE_RESOLVED, 1)
    raw = _page(report, "book-ocr/alpha", 1)
    submission = next(item for item in load_resolved(run) if item.pdf_page == 1)
    figures = [
        (block.block_id, block.region)
        for block in submission.blocks
        if block.kind == "figure" and block.region is not None
    ]
    captions = [
        (block.block_id, block.region)
        for block in submission.blocks
        if block.kind == "caption" and block.region is not None
    ]
    assert nearest_links(figures, captions) == {"p1-b1": "p1-b3"}
    linked = next(block for block in submission.blocks if block.block_id == "p1-b1")
    assert linked.caption_id is None
    assert resolved.caption_association.value == 0.0
    assert raw.caption_association.value == 1.0


def test_a_body_only_resolved_page_leaves_structure_absent(tmp_path: Path) -> None:
    run, reference = caption_case(tmp_path)
    report = score_reference(reference, run)
    bare = _page(report, LANE_RESOLVED, 2)
    raw_bare = _page(report, "book-ocr/alpha", 2)
    assert bare.formula_fidelity.status == STATUS_ABSENT
    assert bare.table_similarity.status == STATUS_ABSENT
    assert bare.figure_overlap.status == STATUS_ABSENT
    assert bare.caption_association.status == STATUS_ABSENT
    assert raw_bare.formula_fidelity.status == STATUS_SCORED
    assert raw_bare.table_similarity.status == STATUS_SCORED
    assert _lane(report, LANE_RESOLVED).by_page_class["bare"].unscorable_fields == {
        FIELD_FORMULA: 1,
        FIELD_TABLE: 1,
        FIELD_FIGURE: 1,
        FIELD_CAPTION: 1,
    }


def test_rendered_formula_and_table_use_the_inventory(tmp_path: Path) -> None:
    run, reference = formula_case(tmp_path)
    page = _page(score_reference(reference, run), LANE_RESOLVED, 1)
    assert page.character_error_rate.value == 0.0
    assert page.formula_fidelity.value == 1.0
    assert page.formula_parse_valid is True
    assert page.table_similarity.value == 1.0
    assert page.printed_page_accuracy.value == 1.0


def test_a_missing_mapping_does_not_search_the_body(tmp_path: Path) -> None:
    run, reference = unmapped_case(tmp_path)
    page = _page(score_reference(reference, run), LANE_RESOLVED, 1)
    assert page.printed_page_accuracy.status == STATUS_ABSENT
    assert page.printed_page_accuracy.value is None
    assert (
        "printed_page_accuracy"
        in _lane(score_reference(reference, run), LANE_RESOLVED).unscorable_fields
    )


def test_score_command_reports_the_resolved_lane_beside_the_engines(tmp_path: Path) -> None:
    run, reference = hand_case(tmp_path)
    result = _RUNNER.invoke(
        app, ["book-ocr", "score", str(run), "--reference", str(reference), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "score" not in payload
    names = [lane["lane"] for lane in payload["lanes"]]
    assert names == ["book-ocr/alpha", "book-ocr/beta", LANE_RESOLVED]
    prose = payload["lanes"][2]["by_page_class"]["prose"]
    assert prose["character_error_rate"]["value"] == 0.5
    assert prose["missing_submissions"] == 1
    assert prose["abstentions"]["symbol"] == 1
    filtered = _RUNNER.invoke(
        app,
        [
            "book-ocr",
            "score",
            str(run),
            "--reference",
            str(reference),
            "--engine",
            "alpha",
        ],
    )
    assert filtered.exit_code == 0, filtered.output
    assert "book-ocr/beta" not in filtered.output
    assert f"score {LANE_RESOLVED}" in filtered.output
    assert "missing 1" in filtered.output
    assert "abstain symbol 1" in filtered.output


def _assert_prose(prose: ClassScore) -> None:
    """Page 1 matches. Page 2 was not submitted, so its three characters stay in the mean."""
    assert prose.pages == 2
    assert prose.missing_submissions == 1
    assert prose.reference_chars == 6
    assert prose.character_error_rate.value == 0.5
    assert prose.word_error_rate.value == 0.5
    assert prose.missing_words == 1
    assert prose.duplicated_words == 0
    assert prose.dangerous_substitution_rates["5/S"] == 0.0
    assert prose.abstentions["symbol"] == 1
    assert prose.abstentions["digit"] == 0
    assert prose.printed_page_accuracy.value == 0.5


def _assert_glyph(glyph: ClassScore) -> None:
    """Reference ``5`` adopted as ``S`` is one substitution and character error 1/1."""
    assert glyph.pages == 1
    assert glyph.missing_submissions == 0
    assert glyph.reference_chars == 1
    assert glyph.character_error_rate.value == 1.0
    assert glyph.word_error_rate.value == 1.0
    assert glyph.missing_words == 1
    assert glyph.duplicated_words == 1
    assert glyph.dangerous_substitutions["5/S"] == 1
    assert glyph.dangerous_substitution_rates["5/S"] == 1.0
    assert sum(glyph.dangerous_substitutions.values()) == 1


def _lane(report: ScoreReport, name: str):
    return next(lane for lane in report.lanes if lane.lane == name)


def _page(report: ScoreReport, name: str, pdf_page: int):
    return next(page for page in _lane(report, name).pages if page.pdf_page == pdf_page)


def _pages(report: ScoreReport) -> dict[tuple[str, int], dict[str, object]]:
    return {
        (lane.lane, page.pdf_page): page.model_dump()
        for lane in report.lanes
        for page in lane.pages
    }


def _changed(
    before: dict[tuple[str, int], dict[str, object]],
    after: dict[tuple[str, int], dict[str, object]],
) -> set[tuple[str, int]]:
    keys = set(before) | set(after)
    return {key for key in keys if before.get(key) != after.get(key)}
