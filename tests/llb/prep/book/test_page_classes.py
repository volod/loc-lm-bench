"""Content signals, class precedence, and seeded pilot selection."""

import numpy as np

from llb.core.contracts.data_prep.book_classes import (
    PAGE_CLASS_LABELS,
    PageClassRecord,
    PlacedSignals,
)
from llb.prep.book.classes.assign import matching_classes
from llb.prep.book.classes.content import measure_content
from llb.prep.book.classes.select import select_pilot

_SHA = "a" * 64


def test_a_ruled_grid_is_a_table_and_prose_lines_are_not():
    table = measure_content(_table())
    prose = measure_content(_prose())
    assert table.ruling_lines >= 10 and table.vertical_rules >= 4
    assert "large_tables" in _classes(table)
    assert prose.ruling_lines < 10
    assert "large_tables" not in _classes(prose)


def test_separated_right_hand_numbers_are_formulas_and_a_full_line_is_not():
    formulas = measure_content(_formulas())
    prose = measure_content(_prose())
    assert formulas.equation_lines >= 3
    assert formulas.formula_density >= 0.06
    assert "dense_formulas" in _classes(formulas)
    assert prose.equation_lines == 0
    assert "dense_formulas" not in _classes(prose)


def test_a_compact_labelled_drawing_is_a_diagram():
    content = measure_content(_diagram())
    assert content.diagram_labels >= 8
    assert content.figure_area >= 0.025
    assert "labelled_diagrams" in _classes(content)


def test_a_short_bottom_rule_with_text_under_it_is_a_footnote():
    content = measure_content(_footnote())
    assert content.footnote_band >= 0.04
    assert "small_footnotes" in _classes(content)
    assert "small_footnotes" in matching_classes(_signals(footnote_band=0.10))
    assert "small_footnotes" not in matching_classes(_signals(footnote_band=0.19))


def test_dot_leaders_are_not_equation_numbers():
    content = measure_content(_leaders())
    assert content.equation_lines < 3
    assert "dense_formulas" not in _classes(content)


def test_glyphs_in_a_corner_and_two_text_columns():
    corners = measure_content(_corners())
    columns = measure_content(_two_columns())
    assert corners.corner_components >= 6
    assert "gutter_corner" in _classes(corners)
    assert columns.column_count >= 2
    assert "multi_column" in _classes(columns)


def test_precedence_keeps_normal_for_an_ordinary_page_and_upside_down_is_not_landscape():
    assert matching_classes(_signals()) == ["normal"]
    curved = matching_classes(_signals(line_curvature_px=4.0, column_count=2))
    assert curved[0] == "severe_curvature" and "multi_column" in curved
    upside = matching_classes(_signals(orientation_deg=180))
    assert upside == ["normal"]
    assert matching_classes(_signals(image_rotation=270))[0] == "landscape"
    wide = matching_classes(_signals(equation_lines=8, formula_density=0.3, column_count=3))
    assert "dense_formulas" not in wide


def test_footnote_selection_prefers_a_rule_near_the_bottom():
    short = [_page(number, ["small_footnotes"], footnote_band=0.06) for number in range(1, 5)]
    tall = [_page(number, ["small_footnotes"], footnote_band=0.14) for number in range(5, 9)]
    picked = select_pilot(short + tall, source_sha256=_SHA, seed=0, size=1)
    assert picked.pages[0].pdf_page <= 4


def test_selection_is_deterministic_and_names_an_empty_class():
    records = [_page(number, ["normal"]) for number in range(1, 13)]
    first = select_pilot(records, source_sha256=_SHA, seed=0, size=3)
    again = select_pilot(records, source_sha256=_SHA, seed=0, size=3)
    other = select_pilot(records, source_sha256=_SHA, seed=1, size=3)
    assert [page.pdf_page for page in first.pages] == [page.pdf_page for page in again.pages]
    assert [page.pdf_page for page in first.pages] != [page.pdf_page for page in other.pages]
    assert PAGE_CLASS_LABELS["severe_curvature"] in first.empty_classes
    assert first.coverage["normal"] == 3
    assert sum(first.coverage.values()) == len(first.pages)


def test_a_class_with_no_free_page_is_named_unfilled_and_a_page_is_chosen_once():
    records = [_page(number, ["normal", "multi_column"], column_count=2) for number in range(1, 4)]
    records.append(_page(4, ["multi_column"], column_count=3))
    selection = select_pilot(records, source_sha256=_SHA, seed=0, size=1)
    assert [page.pdf_page for page in selection.pages] == [4]
    assert selection.pages[0].represents == "multi_column"
    assert PAGE_CLASS_LABELS["normal"] in selection.unfilled_classes
    assert len({page.pdf_page for page in selection.pages}) == len(selection.pages)


def _classes(content: object) -> list[str]:
    return matching_classes(_signals(**_content_fields(content)))


def _content_fields(content: object) -> dict[str, object]:
    names = (
        "formula_density",
        "equation_lines",
        "ruling_lines",
        "vertical_rules",
        "ruled_fraction",
        "figure_area",
        "diagram_labels",
        "column_count",
        "footnote_band",
        "corner_components",
        "gutter_ink",
    )
    return {name: getattr(content, name) for name in names}


def _page(number: int, classes: list[str], **signals: object) -> PageClassRecord:
    return PageClassRecord(
        schema_id="llb.book-ocr-page-classes",
        schema_version="1.0.0",
        kind="book-ocr-page-classes",
        page_id=f"p{number:04d}",
        pdf_page=number,
        classes=classes,
        primary=classes[0],
        signals=_signals(**signals),
    )


def _signals(**overrides: object) -> PlacedSignals:
    values: dict[str, object] = {
        "line_curvature_px": 1.0,
        "gutter_width_px": 0,
        "gutter_side": "none",
        "background_unevenness": 0.1,
        "contrast": 0.8,
        "orientation_deg": 0,
        "image_rotation": 0,
        "page_orientation": "portrait",
        "has_bleed": False,
        "formula_density": 0.0,
        "equation_lines": 0,
        "ruling_lines": 0,
        "vertical_rules": 0,
        "ruled_fraction": 0.0,
        "figure_area": 0.0,
        "diagram_labels": 0,
        "column_count": 1,
        "footnote_band": 0.0,
        "corner_components": 0,
        "gutter_ink": 0.0,
    }
    values.update(overrides)
    return PlacedSignals.model_validate(values)


def _blank(height: int = 480, width: int = 360) -> np.ndarray:
    return np.full((height, width), 230, np.uint8)


def _prose() -> np.ndarray:
    image = _blank()
    for top in range(40, 420, 16):
        image[top : top + 5, 24:336] = 20
    return image


def _table() -> np.ndarray:
    image = _blank()
    for top in range(40, 340, 14):
        image[top : top + 1, 24:336] = 20
    for left in range(40, 320, 40):
        image[40:340, left : left + 1] = 20
    return image


def _formulas() -> np.ndarray:
    image = _prose()
    for top in range(40, 200, 16):
        image[top : top + 5, 40:336] = 230
    for top in range(48, 400, 20):
        image[top : top + 5, 36:140] = 20
        image[top : top + 5, 290:324] = 20
    return image


def _diagram() -> np.ndarray:
    image = _blank()
    for index in range(12):
        image[16:22, 16 + index * 10 : 20 + index * 10] = 20
    image[140:242, 90:92] = 20
    image[140:242, 190:192] = 20
    image[140:142, 90:192] = 20
    image[240:242, 90:192] = 20
    for index in range(10):
        y_pos = 128 + (index % 5) * 22
        x_pos = 78 if index < 5 else 196
        image[y_pos : y_pos + 5, x_pos : x_pos + 4] = 20
    return image


def _leaders() -> np.ndarray:
    image = _formulas()
    for top in range(48, 400, 20):
        for left in range(150, 280, 8):
            image[top : top + 2, left : left + 2] = 190
    return image


def _footnote() -> np.ndarray:
    image = _prose()
    image[430:431, 40:120] = 20
    for index in range(12):
        image[442:447, 40 + index * 8 : 44 + index * 8] = 20
    return image


def _corners() -> np.ndarray:
    image = _prose()
    for index in range(6):
        image[4:8, 4 + index * 5 : 8 + index * 5] = 20
    return image


def _two_columns() -> np.ndarray:
    image = _blank(480, 400)
    for top in range(40, 440, 14):
        image[top : top + 4, 16:170] = 20
        image[top : top + 4, 230:390] = 20
    return image
