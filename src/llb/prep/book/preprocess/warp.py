"""Choose a dewarper and run it. OpenCV is the deterministic path; the others are optional."""

from pathlib import Path
from typing import Any

import numpy as np

from llb.prep.book.preprocess.constants import DEWARP_AUTO, DEWARP_OPENCV, DEWARP_PAGE, DEWARP_UVDOC
from llb.prep.book.preprocess.dewarp import unbend
from llb.prep.book.preprocess.external import Warp, run_page_dewarp, run_uvdoc


def resolve_warp(
    base: np.ndarray, method: str, data_dir: Path, params: dict[str, Any]
) -> Warp | str:
    """The unbend of `base`, or the reason there is no usable one.

    `auto` tries page-dewarp and, when that produces nothing, falls back to OpenCV. The fallback
    is written into `params`. An explicit method that fails is not swapped for another.
    """
    if method == DEWARP_OPENCV:
        return opencv_unbend(base)
    if method == DEWARP_AUTO:
        return _auto(base, data_dir, params)
    if method == DEWARP_PAGE:
        return run_page_dewarp(base, data_dir)
    if method == DEWARP_UVDOC:
        return run_uvdoc(base, data_dir)
    return f"unknown dewarp method {method}"


def opencv_unbend(base: np.ndarray) -> Warp:
    image, map_x, map_y = unbend(base)
    return Warp(image, map_x, map_y)


def _auto(base: np.ndarray, data_dir: Path, params: dict[str, Any]) -> Warp:
    found = run_page_dewarp(base, data_dir)
    if isinstance(found, Warp):
        params["method"] = DEWARP_PAGE
        return found
    params["method"] = DEWARP_OPENCV
    params["fallback_from"] = DEWARP_PAGE
    params["fallback_reason"] = found
    return opencv_unbend(base)
