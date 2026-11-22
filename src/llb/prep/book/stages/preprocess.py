"""The `signals` and `variants` stages.

`signals` measures the page and writes `signals.json`. The runner publishes that file up to
`pages/<pNNNN>/signals.json`. `variants` reads the measurement and writes only the corrections it
calls for, under `pages/<pNNNN>/variants/`, which is this stage's own directory. The original
render is never rewritten.
"""

from collections.abc import Mapping
from typing import Any

from llb.core.paths import resolve_data_dir
from llb.prep.book.preprocess.constants import (
    BACKGROUND_KERNEL,
    BAND_FRACTION,
    BLEED_DENSITY,
    BLEED_EDGE_FRACTION,
    BOW_OUTLIER_RATIO,
    CONTRAST_CORRECT,
    CURVATURE_CORRECT_PX,
    DEWARP_AUTO,
    DEWARP_METHODS,
    DEWARP_OPENCV,
    DEWARP_PAGE,
    DEWARP_UVDOC,
    GUTTER_DARK_FRACTION,
    GUTTER_MAX_FRACTION,
    GUTTER_WIDTH_CORRECT,
    MAX_LINE_BOW_PX,
    MAX_SKEW_DEG,
    NOISE_CORRECT,
    BASELINE_FLIP,
    BASELINE_LINE_GLYPHS,
    BASELINE_MIN_LINES,
    LINE_CENTER_GAP,
    MIN_LINE_SPAN,
    PAPER_LIFT,
    PAPER_PAD,
    POLARITY_MARGIN,
    SIDEWAYS_RATIO,
    SIGNAL_RECIPE,
    SIGNALS_STAGE,
    SKEW_COARSE_STEP,
    SKEW_CORRECT_DEG,
    SKEW_FINE_STEP,
    STROKE_DARK_MAX,
    STROKE_KEEP_FRACTION,
    TILES,
    UNEVENNESS_CORRECT,
    VARIANT_RECIPE,
    VARIANTS_STAGE,
)
from llb.prep.book.preprocess.tooling import page_dewarp_version, uvdoc_version
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE

_DEWARP = "dewarp"


class SignalsStage:
    """One signal vector per rendered page.

    The signal file records the page id, so the cache key includes the PDF page. Identical page
    images do not share one file.
    """

    name = SIGNALS_STAGE
    version = "1"
    inputs: tuple[str, ...] = (RENDER_STAGE,)
    reads_source = True
    model_required = False

    def tool_identity(self) -> str:
        return _opencv_identity()

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return [f"signals: unknown parameter {key!r}" for key in params]

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        resolved = _signal_key()
        resolved["image_rotation"] = inventory.image_rotation
        return resolved

    def run(self, ctx: StageContext) -> StageResult:
        from llb.prep.book.preprocess.images import read_grey
        from llb.prep.book.preprocess.measure import measure_page
        from llb.prep.book.preprocess.records import signals_path, write_signals

        grey = read_grey(ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME)
        signals = measure_page(
            grey, ctx.page_id, ctx.pdf_page, placed_sideways=_placed_sideways(ctx.inventory)
        )
        write_signals(signals_path(ctx.work_dir), signals)
        return StageResult(
            meta={
                "orientation_deg": signals.orientation_deg,
                "skew_deg": signals.skew_deg,
                "line_curvature_px": signals.line_curvature_px,
            }
        )


class VariantsStage:
    """The corrections the signal vector calls for. `dewarp` selects the unbend implementation."""

    name = VARIANTS_STAGE
    version = "1"
    inputs: tuple[str, ...] = (RENDER_STAGE, SIGNALS_STAGE)
    reads_source = False
    model_required = False

    def tool_identity(self) -> str:
        return _opencv_identity()

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        errors = [f"variants: unknown parameter {key!r}" for key in params if key != _DEWARP]
        method = params.get(_DEWARP, DEWARP_OPENCV)
        if method not in DEWARP_METHODS:
            errors.append(f"variants: dewarp must be one of {DEWARP_METHODS}, got {method!r}")
        return errors

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        method = str(params.get(_DEWARP, DEWARP_OPENCV))
        resolved = _variant_key(method)
        _note_dewarper(resolved, method)
        return resolved

    def run(self, ctx: StageContext) -> StageResult:
        from llb.prep.book.preprocess.build import build_variants
        from llb.prep.book.preprocess.images import read_grey
        from llb.prep.book.preprocess.records import read_signals, signals_path

        grey = read_grey(ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME)
        signals = read_signals(signals_path(ctx.inputs[SIGNALS_STAGE]))
        manifest = build_variants(
            grey,
            signals,
            ctx.work_dir,
            method=str(ctx.params[_DEWARP]),
            data_dir=resolve_data_dir(),
        )
        kept = [item.name for item in manifest.variants if item.status == "kept"]
        refused_names = [item.name for item in manifest.variants if item.status == "refused"]
        return StageResult(meta={"kept": kept, "refused": refused_names})


def _placed_sideways(inventory: Any) -> bool:
    """A scan placed at a quarter turn. The renderer has already made the page portrait."""
    return inventory.image_rotation in (90, 270)


def _opencv_identity() -> str:
    import cv2

    return f"opencv {cv2.__version__}"


def _signal_key() -> dict[str, Any]:
    return {
        "recipe": SIGNAL_RECIPE,
        "max_skew_deg": MAX_SKEW_DEG,
        "skew_coarse_step": SKEW_COARSE_STEP,
        "skew_fine_step": SKEW_FINE_STEP,
        "band_fraction": BAND_FRACTION,
        "background_kernel": BACKGROUND_KERNEL,
        "tiles": TILES,
        "polarity_margin": POLARITY_MARGIN,
        "sideways_ratio": SIDEWAYS_RATIO,
        "baseline_flip": BASELINE_FLIP,
        "baseline_min_lines": BASELINE_MIN_LINES,
        "baseline_line_glyphs": BASELINE_LINE_GLYPHS,
        "line_center_gap": LINE_CENTER_GAP,
        "min_line_span": MIN_LINE_SPAN,
        "paper_lift": PAPER_LIFT,
        "paper_pad": PAPER_PAD,
        "gutter_dark_fraction": GUTTER_DARK_FRACTION,
        "bleed_edge_fraction": BLEED_EDGE_FRACTION,
        "bleed_density": BLEED_DENSITY,
    }


def _variant_key(method: str) -> dict[str, Any]:
    return {
        "recipe": VARIANT_RECIPE,
        "dewarp": method,
        "skew_correct_deg": SKEW_CORRECT_DEG,
        "curvature_correct_px": CURVATURE_CORRECT_PX,
        "max_line_bow_px": MAX_LINE_BOW_PX,
        "bow_outlier_ratio": BOW_OUTLIER_RATIO,
        "unevenness_correct": UNEVENNESS_CORRECT,
        "contrast_correct": CONTRAST_CORRECT,
        "gutter_width_correct": GUTTER_WIDTH_CORRECT,
        "gutter_max_fraction": GUTTER_MAX_FRACTION,
        "noise_correct": NOISE_CORRECT,
        "stroke_keep_fraction": STROKE_KEEP_FRACTION,
        "stroke_dark_max": STROKE_DARK_MAX,
    }


def _note_dewarper(resolved: dict[str, Any], method: str) -> None:
    if method in (DEWARP_PAGE, DEWARP_AUTO):
        version = page_dewarp_version()
        if version:
            resolved["page_dewarp"] = version
    if method == DEWARP_UVDOC:
        version = uvdoc_version()
        if version:
            resolved["uvdoc"] = version
