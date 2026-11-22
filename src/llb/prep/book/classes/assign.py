"""Which declared classes a page matches, in the order an engine stage routes on."""

from typing import cast

from llb.core.contracts.data_prep.book_classes import PAGE_CLASS_IDS, PageClassId, PlacedSignals
from llb.prep.book.classes.constants import (
    CORNER_MIN_COMPONENTS,
    DIAGRAM_MAX_LABELS,
    DIAGRAM_MIN_LABELS,
    EQUATION_MIN,
    FIGURE_MIN_AREA,
    FOOTNOTE_MAX_BAND,
    FOOTNOTE_MIN_BAND,
    FORMULA_DENSITY,
    FORMULA_MAX_COLUMNS,
    GREY_CONTRAST,
    GREY_UNEVENNESS,
    MULTI_COLUMN_MIN,
    NORMAL,
    SEVERE_CURVATURE_PX,
    TABLE_MIN_RUN,
    VERTICAL_RULES_MIN,
)

_SIDEWAYS = (90, 270)


def matching_classes(signals: PlacedSignals) -> list[PageClassId]:
    """Every class `signals` meets, in routing order. `normal` only when nothing else matched."""
    matched = [name for name in PAGE_CLASS_IDS if name != NORMAL and _matches(name, signals)]
    return cast(list[PageClassId], matched or [NORMAL])


def _matches(name: str, signals: PlacedSignals) -> bool:
    if name == "severe_curvature":
        return signals.line_curvature_px >= SEVERE_CURVATURE_PX
    if name == "dense_formulas":
        return (
            signals.equation_lines >= EQUATION_MIN
            and signals.column_count <= FORMULA_MAX_COLUMNS
            and signals.formula_density >= FORMULA_DENSITY
        )
    if name == "large_tables":
        return (
            signals.ruling_lines >= TABLE_MIN_RUN and signals.vertical_rules >= VERTICAL_RULES_MIN
        )
    if name == "labelled_diagrams":
        return (
            signals.figure_area >= FIGURE_MIN_AREA
            and DIAGRAM_MIN_LABELS <= signals.diagram_labels <= DIAGRAM_MAX_LABELS
        )
    if name == "small_footnotes":
        return FOOTNOTE_MIN_BAND <= signals.footnote_band <= FOOTNOTE_MAX_BAND
    if name == "gutter_corner":
        return signals.corner_components >= CORNER_MIN_COMPONENTS or signals.has_bleed
    if name == "grey_background":
        return signals.background_unevenness >= GREY_UNEVENNESS or signals.contrast <= GREY_CONTRAST
    if name == "landscape":
        return _landscape(signals)
    if name == "multi_column":
        return signals.column_count >= MULTI_COLUMN_MIN
    return False


def _landscape(signals: PlacedSignals) -> bool:
    """A landscape page box, or a scan placed or measured on its side. Upside down is not this."""
    if signals.page_orientation == "landscape":
        return True
    if signals.image_rotation in _SIDEWAYS:
        return True
    return signals.orientation_deg in _SIDEWAYS
