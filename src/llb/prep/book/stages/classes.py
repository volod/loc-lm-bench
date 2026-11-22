"""The `classes` stage: one class set per rendered page.

It reads the render and the signal vector. The geometric cuts and the content cuts are both in
the cache key, together with the page's placement rotation, so moving a threshold reclassifies
the pages it affects and leaves the render and the signal files untouched.
"""

from collections.abc import Mapping
from typing import Any

from llb.core.contracts.data_prep.book_classes import PageClassRecord, PlacedSignals
from llb.prep.book.classes.assign import matching_classes
from llb.prep.book.classes.constants import (
    CLASS_SCHEMA_ID,
    CLASSES_STAGE,
    SCHEMA_VERSION,
    class_thresholds,
)
from llb.prep.book.classes.records import classes_path, write_classes
from llb.prep.book.preprocess.constants import SIGNALS_STAGE
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE


class ClassesStage:
    """Every declared class a page matches, and the one routing uses."""

    name = CLASSES_STAGE
    version = "1"
    inputs: tuple[str, ...] = (RENDER_STAGE, SIGNALS_STAGE)
    reads_source = False
    model_required = False

    def tool_identity(self) -> str:
        import cv2

        return f"opencv {cv2.__version__}"

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return [f"classes: unknown parameter {key!r}" for key in params]

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        resolved = class_thresholds()
        resolved["image_rotation"] = inventory.image_rotation
        resolved["page_orientation"] = inventory.orientation
        return resolved

    def run(self, ctx: StageContext) -> StageResult:
        from llb.prep.book.classes.content import measure_content
        from llb.prep.book.preprocess.images import read_grey
        from llb.prep.book.preprocess.records import read_signals, signals_path

        grey = read_grey(ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME)
        signals = read_signals(signals_path(ctx.inputs[SIGNALS_STAGE]))
        content = measure_content(grey, signals.orientation_deg)
        placed = _placed(signals, content, ctx.inventory)
        classes = matching_classes(placed)
        record = PageClassRecord(
            schema_id=CLASS_SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            kind="book-ocr-page-classes",
            page_id=ctx.page_id,
            pdf_page=ctx.pdf_page,
            classes=classes,
            primary=classes[0],
            signals=placed,
        )
        write_classes(classes_path(ctx.work_dir), record)
        return StageResult(meta={"primary": classes[0], "classes": list(classes)})


def _placed(signals: Any, content: Any, inventory: Any) -> PlacedSignals:
    return PlacedSignals(
        line_curvature_px=signals.line_curvature_px,
        gutter_width_px=signals.gutter_width_px,
        gutter_side=signals.gutter_side,
        background_unevenness=signals.background_unevenness,
        contrast=signals.contrast,
        orientation_deg=signals.orientation_deg,
        image_rotation=inventory.image_rotation,
        page_orientation=inventory.orientation,
        has_bleed=signals.bleed is not None,
        formula_density=content.formula_density,
        equation_lines=content.equation_lines,
        ruling_lines=content.ruling_lines,
        vertical_rules=content.vertical_rules,
        ruled_fraction=content.ruled_fraction,
        figure_area=content.figure_area,
        diagram_labels=content.diagram_labels,
        column_count=content.column_count,
        footnote_band=content.footnote_band,
        corner_components=content.corner_components,
        gutter_ink=content.gutter_ink,
    )
