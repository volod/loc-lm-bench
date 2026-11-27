"""Pad a figure and crop it from the best kept geometric variant.

The recorded box stays in the original render. The pixels come from dewarp
when that variant was kept, otherwise deskew, otherwise a quarter-turn, and
otherwise the render itself.
"""

from pathlib import Path

from llb.core.contracts.data_prep.book_preprocess import (
    AffineTransform,
    GridTransform,
    Transform,
    VariantManifest,
    VariantRecord,
)
from llb.prep.book.assets.boxes import Box, center, clip, pixel_box, span_limit, unite
from llb.prep.book.assets.constants import (
    GEOMETRIC_ORDER,
    MARGIN_FRAC,
    MARGIN_MAX_PX,
    MARGIN_MIN_PX,
    VARIANT_SAMPLE_STEP,
)
from llb.prep.book.cache import page_dir
from llb.prep.book.engines.image import png_size
from llb.prep.book.preprocess.constants import MANIFEST_FILENAME, VARIANTS_STAGE
from llb.prep.book.preprocess.geometry import Affine
from llb.prep.book.preprocess.maps import locate_affine, map_point
from llb.prep.book.preprocess.records import read_manifest
from llb.prep.book.render.publish import Crop

_Crop = tuple[Crop, str, tuple[int, int, int, int]]


def padded_figure(
    union: Box, width: int, height: int, caption: Box | None, others: list[Box]
) -> Box:
    """The crop in the original frame: margin, then the caption, then the page."""
    margin = span_limit(width, height, MARGIN_FRAC, MARGIN_MIN_PX, MARGIN_MAX_PX)
    grown = Box(union.x0 - margin, union.y0 - margin, union.x1 + margin, union.y1 + margin)
    grown = _avoid(grown, union, others)
    if caption is not None:
        grown = unite((grown, caption))
    return clip(grown, width, height)


def _avoid(grown: Box, union: Box, others: list[Box]) -> Box:
    """Pull back a side whose margin contains another figure's center."""
    x0, y0, x1, y1 = grown.x0, grown.y0, grown.x1, grown.y1
    for other in others:
        cx, cy = center(other)
        if not (grown.x0 <= cx <= grown.x1 and grown.y0 <= cy <= grown.y1):
            continue
        if cx < union.x0:
            x0 = union.x0
        elif cx > union.x1:
            x1 = union.x1
        if cy < union.y0:
            y0 = union.y0
        elif cy > union.y1:
            y1 = union.y1
    return Box(x0, y0, x1, y1)


def crop_figure(run_dir: Path, page_id: str, original: Path | None, box: Box) -> _Crop | None:
    """Pixels for ``box``. The third value is the original-frame pixel box."""
    if original is None or not original.is_file():
        return None
    src_width, src_height = png_size(original)
    recorded = pixel_box(box, src_width, src_height)
    if recorded is None:
        return None
    variant = _variant_crop(run_dir, page_id, box)
    if variant is not None:
        path, pixels, name = variant
        return Crop(path, pixels), name, recorded
    return Crop(original, recorded), "original", recorded


def best_geometric(manifest: VariantManifest) -> VariantRecord | None:
    """The most corrected kept geometric variant, or none."""
    found = {item.name: item for item in manifest.variants}
    for name in GEOMETRIC_ORDER:
        item = found.get(name)
        if item is not None and _usable(item):
            return item
    return None


def original_box_to_variant(transform: Transform, box: Box) -> Box | None:
    """``box`` in original pixels, expressed in the variant image."""
    if isinstance(transform, AffineTransform):
        return _affine_box(transform, box)
    return _grid_box(transform, box)


def _usable(item: VariantRecord) -> bool:
    return (
        item.status == "kept" and item.geometric and bool(item.image) and item.transform is not None
    )


def _variant_crop(
    run_dir: Path, page_id: str, box: Box
) -> tuple[Path, tuple[int, int, int, int], str] | None:
    directory = page_dir(run_dir, page_id) / VARIANTS_STAGE
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    record = best_geometric(read_manifest(manifest_path))
    if record is None or not record.image or record.transform is None:
        return None
    path = directory / record.image
    if not path.is_file():
        return None
    mapped = original_box_to_variant(record.transform, box)
    if mapped is None:
        return None
    width, height = png_size(path)
    pixels = pixel_box(mapped, width, height)
    if pixels is None:
        return None
    return path, pixels, record.name


def _affine_box(transform: AffineTransform, box: Box) -> Box | None:
    locate = locate_affine(_affine(transform))
    points: list[tuple[float, float]] = []
    for x, y in _samples(box):
        hit = locate(x, y)
        if hit is not None:
            points.append(hit)
    if len(points) < 2:
        return None
    return Box(
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _affine(transform: AffineTransform) -> Affine:
    first, second = transform.matrix
    row0 = (float(first[0]), float(first[1]), float(first[2]))
    row1 = (float(second[0]), float(second[1]), float(second[2]))
    return Affine(
        (row0, row1), transform.width, transform.height, transform.src_width, transform.src_height
    )


def _samples(box: Box) -> tuple[tuple[float, float], ...]:
    xs = (box.x0, (box.x0 + box.x1) / 2.0, box.x1)
    ys = (box.y0, (box.y0 + box.y1) / 2.0, box.y1)
    return tuple((x, y) for y in ys for x in xs)


def _grid_box(transform: GridTransform, box: Box) -> Box | None:
    step = VARIANT_SAMPLE_STEP
    hits = [(x, y) for y in _axis(transform.height, step) for x in _axis(transform.width, step)]
    inside = [(x, y) for x, y in hits if _lands_inside(transform, box, x, y)]
    if not inside:
        return None
    return Box(
        min(point[0] for point in inside) - step,
        min(point[1] for point in inside) - step,
        max(point[0] for point in inside) + step,
        max(point[1] for point in inside) + step,
    )


def _axis(length: int, step: int) -> list[int]:
    found = list(range(0, max(length, 1), step))
    last = max(length - 1, 0)
    if not found or found[-1] != last:
        found.append(last)
    return found


def _lands_inside(transform: GridTransform, box: Box, x: int, y: int) -> bool:
    origin_x, origin_y = map_point(transform, float(x), float(y))
    return box.x0 <= origin_x <= box.x1 and box.y0 <= origin_y <= box.y1
