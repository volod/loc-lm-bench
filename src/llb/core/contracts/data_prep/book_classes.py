"""Per-page class set and the pilot selection drawn from it.

The class stage writes one record per page. Pilot selection copies the pages it chose, with the
signals that placed them, and never a page image or a recovered transcription.
"""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.book import ImageRotation
from llb.core.contracts.data_prep.book_preprocess import GutterSide, Orientation
from llb.core.contracts.data_prep.corpus import DataPrepRow

# Routing order. The first class a page matches is the one `by_page_class` sees; `normal` is the
# residual, present only when nothing else matched. A later class can still be why a pilot page
# was chosen.
PAGE_CLASS_IDS: tuple[str, ...] = (
    "severe_curvature",
    "dense_formulas",
    "large_tables",
    "labelled_diagrams",
    "small_footnotes",
    "gutter_corner",
    "grey_background",
    "landscape",
    "multi_column",
    "normal",
)
PAGE_CLASS_LABELS: dict[str, str] = {
    "severe_curvature": "severe curvature",
    "dense_formulas": "dense formulas",
    "large_tables": "large or irregular tables",
    "labelled_diagrams": "labelled diagrams",
    "small_footnotes": "small footnotes",
    "gutter_corner": "gutter and corner text",
    "grey_background": "grey or shadowed background",
    "landscape": "landscape",
    "multi_column": "multi-column layout",
    "normal": "normal",
}
PageClassId = Literal[
    "severe_curvature",
    "dense_formulas",
    "large_tables",
    "labelled_diagrams",
    "small_footnotes",
    "gutter_corner",
    "grey_background",
    "landscape",
    "multi_column",
    "normal",
]
_CLASS_ID = PageClassId
_SHA256 = r"^[0-9a-f]{64}$"


class PlacedSignals(DataPrepRow):
    """The measurements that decided a page's classes. No pixels and no text."""

    line_curvature_px: float = Field(ge=0)
    gutter_width_px: int = Field(ge=0)
    gutter_side: GutterSide
    background_unevenness: float = Field(ge=0)
    contrast: float = Field(ge=0)
    orientation_deg: Orientation
    image_rotation: ImageRotation | None = None
    page_orientation: Literal["portrait", "landscape"]
    has_bleed: bool
    formula_density: float = Field(ge=0, le=1)
    equation_lines: int = Field(ge=0)
    ruling_lines: int = Field(ge=0)
    vertical_rules: int = Field(ge=0)
    ruled_fraction: float = Field(ge=0, le=1)
    figure_area: float = Field(ge=0, le=1)
    diagram_labels: int = Field(ge=0)
    column_count: int = Field(ge=1, le=8)
    footnote_band: float = Field(ge=0, le=1)
    corner_components: int = Field(ge=0)
    gutter_ink: float = Field(ge=0, le=1)


class PageClassRecord(ArtifactContract):
    """`pages/<pNNNN>/classes.json`: every class the page matches, and the one routing uses."""

    schema_id: Literal["llb.book-ocr-page-classes"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-page-classes"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    classes: list[_CLASS_ID] = Field(min_length=1)
    primary: _CLASS_ID
    signals: PlacedSignals

    @model_validator(mode="after")
    def _primary_is_first(self) -> "PageClassRecord":
        if self.primary != self.classes[0]:
            raise ValueError("primary class is the first entry of classes")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must not repeat")
        return self


class PilotPageEntry(DataPrepRow):
    """One pilot page: the class it was chosen to represent, and the signals that placed it."""

    pdf_page: int = Field(ge=1)
    page_id: str = Field(min_length=1)
    represents: _CLASS_ID
    classes: list[_CLASS_ID] = Field(min_length=1)
    primary: _CLASS_ID
    signals: PlacedSignals

    @model_validator(mode="after")
    def _represents_a_match(self) -> "PilotPageEntry":
        if self.represents not in self.classes:
            raise ValueError("a pilot page must represent a class it matches")
        if self.primary != self.classes[0]:
            raise ValueError("primary class is the first entry of classes")
        return self


class PilotSelection(ArtifactContract):
    """The seeded pilot set. `empty_classes` names every declared class no page matched."""

    schema_id: Literal["llb.book-ocr-pilot-selection"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-pilot-selection"]
    source_sha256: str = Field(pattern=_SHA256)
    seed: int = Field(ge=0)
    size: int = Field(ge=1)
    pages: list[PilotPageEntry]
    empty_classes: list[str]
    unfilled_classes: list[str]
    coverage: dict[str, int]

    @model_validator(mode="after")
    def _coverage_matches_pages(self) -> "PilotSelection":
        counts: dict[str, int] = {name: 0 for name in PAGE_CLASS_IDS}
        seen: set[int] = set()
        for page in self.pages:
            if page.pdf_page in seen:
                raise ValueError(f"pdf page {page.pdf_page} is selected twice")
            seen.add(page.pdf_page)
            counts[page.represents] += 1
        if dict(self.coverage) != counts:
            raise ValueError("coverage must count the selected pages per class")
        return self
