"""Page crops a formula or a table keeps beside its rendered form."""

from pathlib import Path

from llb.prep.book.engines.image import png_size
from llb.prep.book.render.names import chapter_slug, figure_asset, parents_of
from llb.prep.book.render.publish import Asset, Crop
from llb.prep.book.render.segment import Span
from llb.prep.book.render.types import BookLoad, RenderBlock
from llb.prep.book.review.constants import CROP_PADDING_PX
from llb.prep.book.review.crops import padded_box

Paths = dict[str, str]


def technical_assets(
    spans: list[Span], loaded: BookLoad, book: str
) -> tuple[Paths, Paths, Paths, dict[str, Asset]]:
    """Formula crops, unfaithful table images, disputed-cell crops, and their bytes."""
    parents = parents_of(loaded.structure.hierarchy)
    formulas: Paths = {}
    images: Paths = {}
    evidence: Paths = {}
    blobs: dict[str, Asset] = {}
    for span in spans:
        chapter = chapter_slug(span.node, parents, book)
        for block in span.blocks:
            _one(block, chapter, loaded, formulas, images, evidence, blobs)
    return formulas, images, evidence, blobs


def _one(
    block: RenderBlock,
    chapter: str,
    loaded: BookLoad,
    formulas: Paths,
    images: Paths,
    evidence: Paths,
    blobs: dict[str, Asset],
) -> None:
    image = loaded.images.get(block.pdf_page)
    if block.region is None or image is None:
        return
    if block.formula_crop:
        _put(formulas, blobs, block, image, f"review/{block.block_id}-formula.png", CROP_PADDING_PX)
    if block.table_image:
        rel = figure_asset(chapter, block.page_id, block.ordinal, "table", "")
        _put(images, blobs, block, image, rel, 0)
    if block.table_evidence:
        _put(evidence, blobs, block, image, f"review/{block.block_id}-table.png", CROP_PADDING_PX)


def _put(
    paths: Paths,
    blobs: dict[str, Asset],
    block: RenderBlock,
    image: Path,
    rel: str,
    padding: int,
) -> None:
    if block.region is None:
        return
    width, height = png_size(image)
    box = padded_box(block.region, (height, width), padding)
    if box is None:
        return
    paths[block.block_id] = rel
    blobs[rel] = Crop(image, box)
