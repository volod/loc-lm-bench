"""Thresholds for the page-class stage.

A page matches a class when its signal crosses the threshold named here. `normal` is what remains.
The numbers are part of the class stage's cache key, so moving one reclassifies the pages it
affects and nothing else.
"""

from typing import Final

from llb.core.contracts.data_prep.book_classes import PAGE_CLASS_IDS, PAGE_CLASS_LABELS

CLASS_SCHEMA_ID: Final = "llb.book-ocr-page-classes"
PILOT_SCHEMA_ID: Final = "llb.book-ocr-pilot-selection"
SCHEMA_VERSION: Final = "1.0.0"
CLASSES_FILENAME: Final = "classes.json"
CLASSES_STAGE: Final = "classes"
PILOT_DIRNAME: Final = "pilot"
SELECTION_FILENAME: Final = "selection.json"
CLASS_RECIPE: Final = "1"
ANALYSIS_LONG_SIDE: Final = 720

# Geometric. Severe curvature is the same bow the dewarp gate treats as worth unbending.
SEVERE_CURVATURE_PX: Final = 3.0
# Grey or shadowed is the upper tail of background spread, not the mild falloff almost every
# scan of this book carries. The contrast gate catches a page whose ink and paper nearly meet.
GREY_UNEVENNESS: Final = 0.20
GREY_CONTRAST: Final = 0.35

# Content. A plain two-column page stays below these. `formula_density` is stored as 0 when a page
# has fewer than FORMULA_MIN_LINES text lines, so the density cut is not applied to a short page.
FORMULA_DENSITY: Final = 0.06
FORMULA_MIN_LINES: Final = 12
EQUATION_MIN: Final = 3
FORMULA_MAX_COLUMNS: Final = 2
# A gap this full of faint marks is a dot leader, not the space before an equation number.
FORMULA_LEADER_GAP: Final = 0.05
TABLE_MIN_RUN: Final = 10
VERTICAL_RULES_MIN: Final = 4
FIGURE_MIN_AREA: Final = 0.025
DIAGRAM_MIN_LABELS: Final = 8
DIAGRAM_MAX_LABELS: Final = 70
FOOTNOTE_MIN_BAND: Final = 0.04
# A rule at the top of the bottom fifth is a separator. A footnote rule sits lower.
FOOTNOTE_MAX_BAND: Final = 0.15
CORNER_MIN_COMPONENTS: Final = 6
MULTI_COLUMN_MIN: Final = 2

NORMAL: Final = "normal"

__all__ = [
    "ANALYSIS_LONG_SIDE",
    "CLASS_RECIPE",
    "CLASS_SCHEMA_ID",
    "class_thresholds",
    "CLASSES_FILENAME",
    "CLASSES_STAGE",
    "CORNER_MIN_COMPONENTS",
    "DIAGRAM_MAX_LABELS",
    "DIAGRAM_MIN_LABELS",
    "EQUATION_MIN",
    "FIGURE_MIN_AREA",
    "FOOTNOTE_MIN_BAND",
    "FOOTNOTE_MAX_BAND",
    "FORMULA_DENSITY",
    "FORMULA_LEADER_GAP",
    "FORMULA_MAX_COLUMNS",
    "FORMULA_MIN_LINES",
    "GREY_CONTRAST",
    "GREY_UNEVENNESS",
    "MULTI_COLUMN_MIN",
    "NORMAL",
    "PAGE_CLASS_IDS",
    "PAGE_CLASS_LABELS",
    "PILOT_DIRNAME",
    "PILOT_SCHEMA_ID",
    "SCHEMA_VERSION",
    "SELECTION_FILENAME",
    "SEVERE_CURVATURE_PX",
    "TABLE_MIN_RUN",
    "VERTICAL_RULES_MIN",
]


def class_thresholds() -> dict[str, float | int | str]:
    """The cuts the class stage applies. They are part of its cache key."""
    return {
        "recipe": CLASS_RECIPE,
        "severe_curvature_px": SEVERE_CURVATURE_PX,
        "grey_unevenness": GREY_UNEVENNESS,
        "grey_contrast": GREY_CONTRAST,
        "formula_density": FORMULA_DENSITY,
        "formula_min_lines": FORMULA_MIN_LINES,
        "equation_min": EQUATION_MIN,
        "formula_max_columns": FORMULA_MAX_COLUMNS,
        "formula_leader_gap": FORMULA_LEADER_GAP,
        "table_min_run": TABLE_MIN_RUN,
        "vertical_rules_min": VERTICAL_RULES_MIN,
        "figure_min_area": FIGURE_MIN_AREA,
        "diagram_min_labels": DIAGRAM_MIN_LABELS,
        "diagram_max_labels": DIAGRAM_MAX_LABELS,
        "footnote_min_band": FOOTNOTE_MIN_BAND,
        "footnote_max_band": FOOTNOTE_MAX_BAND,
        "corner_min": CORNER_MIN_COMPONENTS,
        "multi_column_min": MULTI_COLUMN_MIN,
    }
