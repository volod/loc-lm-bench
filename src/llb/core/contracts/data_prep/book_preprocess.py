"""Page signals and the variant manifest a `book-ocr` preprocess stage writes.

Signals are measurements, not decisions. The variant manifest is the decision: which corrections
the signals called for, which image was kept, and -- for every geometric variant -- the point map
that sends a pixel back to the original render. A later stage that reads a box off a rotated or
dewarped image uses that map, so the block ledger stays in one frame.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.corpus import DataPrepRow

Orientation = Literal[0, 90, 180, 270]
GutterSide = Literal["left", "right", "top", "bottom", "none"]
BleedSide = Literal["left", "right", "top", "bottom"]
VariantStatus = Literal["kept", "refused"]


class BleedBand(DataPrepRow):
    """A neighbour-page bleed, as a region of the original render. It is not cropped."""

    side: BleedSide
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int = Field(ge=1)
    y1: int = Field(ge=1)

    @model_validator(mode="after")
    def _positive_area(self) -> "BleedBand":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bleed band must have positive area")
        return self


class PageSignals(ArtifactContract):
    """`pages/<pNNNN>/signals.json`: one page's measured signal vector."""

    schema_id: Literal["llb.book-ocr-page-signals"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-page-signals"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    skew_deg: float
    line_curvature_px: float = Field(ge=0)
    background_unevenness: float = Field(ge=0)
    contrast: float = Field(ge=0)
    blur: float = Field(ge=0)
    noise: float = Field(ge=0)
    gutter_side: GutterSide
    gutter_width_px: int = Field(ge=0)
    bleed: BleedBand | None = None
    orientation_deg: Orientation


class AffineTransform(DataPrepRow):
    """Variant pixel to original pixel: `[x, y, 1]` times this 2x3 matrix."""

    kind: Literal["affine"]
    matrix: list[list[float]]
    src_width: int = Field(ge=1)
    src_height: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)

    @model_validator(mode="after")
    def _matrix_shape(self) -> "AffineTransform":
        if len(self.matrix) != 2 or any(len(row) != 3 for row in self.matrix):
            raise ValueError("affine matrix must be 2x3")
        return self


class GridTransform(DataPrepRow):
    """A sampled point map. `rows`/`cols` are variant pixels; `map_x`/`map_y` are original pixels."""

    kind: Literal["grid"]
    rows: list[int] = Field(min_length=2)
    cols: list[int] = Field(min_length=2)
    src_width: int = Field(ge=1)
    src_height: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    map_x: list[list[float]]
    map_y: list[list[float]]

    @model_validator(mode="after")
    def _grid_shape(self) -> "GridTransform":
        if len(self.map_x) != len(self.rows) or len(self.map_y) != len(self.rows):
            raise ValueError("grid map row count must match rows")
        wide = len(self.cols)
        if any(len(row) != wide for row in self.map_x) or any(
            len(row) != wide for row in self.map_y
        ):
            raise ValueError("grid map column count must match cols")
        return self


Transform = Annotated[AffineTransform | GridTransform, Field(discriminator="kind")]


class VariantRecord(DataPrepRow):
    """One image variant, or the reason it was refused and not written."""

    name: str = Field(min_length=1)
    status: VariantStatus
    reason: str | None = None
    parameters: JsonObject = Field(default_factory=dict)
    image: str | None = None
    geometric: bool
    transform: Transform | None = None


class VariantManifest(ArtifactContract):
    """`pages/<pNNNN>/variants/manifest.json`: the variants generated for one page."""

    schema_id: Literal["llb.book-ocr-variant-manifest"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-variant-manifest"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    variants: list[VariantRecord] = Field(min_length=1)
