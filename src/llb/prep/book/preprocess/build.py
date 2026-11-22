"""Generate the image variants one page's signals call for, and record why the rest were skipped.

Photometric variants are applied to the original render, so their map is the identity. Geometric
variants are composed outward: orientation, then deskew, then unbend. Each stored map sends a
variant pixel back to the original render. A variant the guards refuse is recorded and not written.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from llb.core.contracts.data_prep.book_preprocess import PageSignals, VariantManifest, VariantRecord
from llb.prep.book.preprocess.constants import (
    CONTRAST,
    DENOISE,
    DESKEW,
    DEWARP,
    GUTTER,
    ILLUMINATION,
    ORIGINAL,
    ROTATED,
    SKEW_CORRECT_DEG,
    VARIANT_IMAGE,
)
from llb.prep.book.preprocess.decide import selected
from llb.prep.book.preprocess.geometry import (
    Affine,
    affine_covers,
    compose,
    deskew,
    grid_covers,
    identity,
    push_through,
    rotate_upright,
)
from llb.prep.book.preprocess.guards import refusal
from llb.prep.book.preprocess.images import write_png
from llb.prep.book.preprocess.maps import Locate, locate_affine, locate_dense
from llb.prep.book.preprocess.photometric import (
    denoise,
    flatten_background,
    lift_gutter,
    local_contrast,
)
from llb.prep.book.preprocess.records import (
    affine_record,
    grid_record,
    kept,
    manifest_path,
    page_manifest,
    refused,
    write_manifest,
)
from llb.prep.book.preprocess.warp import resolve_warp

log = logging.getLogger(__name__)


def build_variants(
    grey: np.ndarray,
    signals: PageSignals,
    out_dir: Path,
    *,
    method: str,
    data_dir: Path,
) -> VariantManifest:
    """Write the variants `signals` calls for under `out_dir` and return their manifest."""
    choice = selected(signals)
    records = [_original(grey)]
    records.extend(_geometric(grey, signals, choice, out_dir, method, data_dir))
    records.extend(_photometric(grey, signals, choice, out_dir))
    manifest = page_manifest(signals.page_id, signals.pdf_page, records)
    write_manifest(manifest_path(out_dir), manifest)
    return manifest


def _original(grey: np.ndarray) -> VariantRecord:
    height, width = grey.shape[:2]
    return kept(ORIGINAL, None, affine_record(identity(width, height)), {}, geometric=False)


def _geometric(
    grey: np.ndarray,
    signals: PageSignals,
    choice: dict[str, bool],
    out_dir: Path,
    method: str,
    data_dir: Path,
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    if choice[ROTATED]:
        records.append(_rotated(grey, signals, out_dir))
    if choice[DESKEW]:
        records.append(_deskewed(grey, signals, out_dir))
    if choice[DEWARP]:
        records.append(_dewarped(grey, signals, out_dir, method, data_dir))
    return records


def _photometric(
    grey: np.ndarray, signals: PageSignals, choice: dict[str, bool], out_dir: Path
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    if choice[ILLUMINATION]:
        image = flatten_background(grey)
        records.append(_tone(ILLUMINATION, image, grey, out_dir, _uneven(signals)))
    if choice[CONTRAST]:
        records.append(_tone(CONTRAST, local_contrast(grey), grey, out_dir, _contrast(signals)))
    if choice[GUTTER]:
        records.append(_gutter(grey, signals, out_dir))
    if choice[DENOISE]:
        records.append(_tone(DENOISE, denoise(grey), grey, out_dir, {"noise": signals.noise}))
    return records


def _rotated(grey: np.ndarray, signals: PageSignals, out_dir: Path) -> VariantRecord:
    image, transform, params = _prepare(grey, signals, rotate=True, skew=False)
    return _affine_variant(ROTATED, image, transform, grey, out_dir, params)


def _deskewed(grey: np.ndarray, signals: PageSignals, out_dir: Path) -> VariantRecord:
    image, transform, params = _prepare(grey, signals, rotate=True, skew=True)
    return _affine_variant(DESKEW, image, transform, grey, out_dir, params)


def _dewarped(
    grey: np.ndarray, signals: PageSignals, out_dir: Path, method: str, data_dir: Path
) -> VariantRecord:
    skew = abs(signals.skew_deg) >= SKEW_CORRECT_DEG
    base, base_map, params = _prepare(grey, signals, rotate=True, skew=skew)
    params["curvature_px"] = signals.line_curvature_px
    params["method"] = method
    warped = resolve_warp(base, method, data_dir, params)
    if isinstance(warped, str):
        return _refuse(DEWARP, warped, params)
    map_x, map_y = warped.map_x, warped.map_y
    if not _is_identity(base_map):
        map_x, map_y = push_through(map_x, map_y, base_map)
    return _grid_variant(DEWARP, warped.image, map_x, map_y, grey, out_dir, params)


def _gutter(grey: np.ndarray, signals: PageSignals, out_dir: Path) -> VariantRecord:
    lifted = lift_gutter(grey, signals.gutter_side, signals.gutter_width_px)
    params: dict[str, Any] = {
        "gutter_side": signals.gutter_side,
        "gutter_width_px": signals.gutter_width_px,
    }
    return _tone(GUTTER, lifted, grey, out_dir, params)


def _prepare(
    grey: np.ndarray, signals: PageSignals, *, rotate: bool, skew: bool
) -> tuple[np.ndarray, Affine, dict[str, Any]]:
    height, width = grey.shape[:2]
    image: np.ndarray = grey
    transform = identity(width, height)
    params: dict[str, Any] = {}
    if rotate and signals.orientation_deg:
        image, rotation = rotate_upright(image, signals.orientation_deg)
        transform = rotation
        params["orientation_deg"] = signals.orientation_deg
    if skew:
        image, leveled = deskew(image, signals.skew_deg)
        transform = compose(transform, leveled)
        params["skew_deg"] = signals.skew_deg
    return image, transform, params


def _affine_variant(
    name: str,
    image: np.ndarray,
    transform: Affine,
    original: np.ndarray,
    out_dir: Path,
    params: dict[str, Any],
) -> VariantRecord:
    reason = refusal(
        original, image, locate_affine(transform), geometric=True, covered=affine_covers(transform)
    )
    if reason:
        return _refuse(name, reason, params)
    relative = f"{name}/{VARIANT_IMAGE}"
    write_png(out_dir / relative, image)
    return kept(name, relative, affine_record(transform), params, geometric=True)


def _grid_variant(
    name: str,
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    original: np.ndarray,
    out_dir: Path,
    params: dict[str, Any],
) -> VariantRecord:
    if image.shape[:2] != map_y.shape:
        return _refuse(name, "point map does not match the dewarped image", params)
    height, width = original.shape[:2]
    locate = locate_dense(map_x, map_y, width, height)
    reason = refusal(
        original, image, locate, geometric=True, covered=grid_covers(map_x, map_y, width, height)
    )
    if reason:
        return _refuse(name, reason, params)
    relative = f"{name}/{VARIANT_IMAGE}"
    write_png(out_dir / relative, image)
    return kept(name, relative, grid_record(map_x, map_y, width, height), params, geometric=True)


def _tone(
    name: str, image: np.ndarray, original: np.ndarray, out_dir: Path, params: dict[str, Any]
) -> VariantRecord:
    height, width = original.shape[:2]
    reason = refusal(original, image, _here(width, height), geometric=False, covered=True)
    if reason:
        return _refuse(name, reason, params)
    relative = f"{name}/{VARIANT_IMAGE}"
    write_png(out_dir / relative, image)
    return kept(name, relative, affine_record(identity(width, height)), params, geometric=False)


def _refuse(name: str, reason: str, params: dict[str, Any]) -> VariantRecord:
    log.info("variant %s refused: %s", name, reason)
    return refused(name, reason, params, geometric=name in (ROTATED, DESKEW, DEWARP))


def _here(width: int, height: int) -> Locate:
    def locate(x: float, y: float) -> tuple[float, float] | None:
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return x, y

    return locate


def _is_identity(transform: Affine) -> bool:
    same_size = transform.width == transform.src_width and transform.height == transform.src_height
    return same_size and transform.matrix == identity(1, 1).matrix


def _uneven(signals: PageSignals) -> dict[str, Any]:
    return {"background_unevenness": signals.background_unevenness}


def _contrast(signals: PageSignals) -> dict[str, Any]:
    return {"contrast": signals.contrast}
