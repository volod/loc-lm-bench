"""Choose the figure assets a vault embeds.

Each asset is one drawing: the pieces of a split figure, the printed caption
when the page makes that association, and a crop from the best geometric
variant. Repeated badges and isolated specks stay out of ``assets/``.
"""

from pathlib import Path

from llb.core.contracts.data_prep.book_structure import DocumentStructure
from llb.prep.book.assets.boxes import Box, from_region
from llb.prep.book.assets.captions import assign_proximity, caption_in_crop
from llb.prep.book.assets.printed import assign_printed
from llb.prep.book.assets.constants import REASON_NOISE, REASON_ORNAMENT
from llb.prep.book.assets.crop import crop_figure, padded_figure
from llb.prep.book.assets.record import figure_record, figure_row, rejected_row
from llb.prep.book.assets.select import drop_noise, group_pieces, repeated_ornament_ids
from llb.prep.book.assets.types import FigureGroup, FigureRender, Piece, piece_key
from llb.prep.book.render.names import chapter_slug, figure_asset, parents_of
from llb.prep.book.render.publish import Asset, Crop
from llb.prep.book.render.segment import Span
from llb.prep.book.render.types import BookLoad, RenderBlock


def figure_assets(spans: list[Span], loaded: BookLoad, book: str, run_dir: Path) -> FigureRender:
    """Assets, caption text, suppressed blocks, ornament relabels, and the region record."""
    pieces, captions = _collect(spans, book, loaded.structure)
    ornaments = repeated_ornament_ids(pieces)
    kept = [piece for piece in pieces if piece.block.block_id not in ornaments]
    assign_printed(kept, captions, loaded.structure.inventory)
    groups = group_pieces(kept)
    assign_proximity(groups, captions)
    groups, noise = drop_noise(groups)
    paths: dict[str, str] = {}
    blobs: dict[str, Asset] = {}
    caption_text: dict[str, str] = {}
    suppressed: set[str] = set()
    rows: list[dict[str, object]] = []
    for group in groups:
        _emit(
            group, groups, captions, loaded, run_dir, paths, blobs, caption_text, suppressed, rows
        )
    rejected = _rejected(pieces, ornaments, noise)
    record = figure_record(rows, rejected)
    labelled = ornaments | noise
    return FigureRender(
        paths, blobs, caption_text, frozenset(suppressed), frozenset(labelled), record
    )


def _collect(
    spans: list[Span], book: str, structure: DocumentStructure
) -> tuple[list[Piece], list[RenderBlock]]:
    parents = parents_of(structure.hierarchy)
    pieces: list[Piece] = []
    captions: list[RenderBlock] = []
    for span in spans:
        chapter = chapter_slug(span.node, parents, book)
        for block in span.blocks:
            _take(pieces, captions, block, chapter)
    pieces.sort(key=piece_key)
    captions.sort(
        key=lambda block: (block.pdf_page, block.reading_order, block.ordinal, block.block_id)
    )
    return pieces, captions


def _take(
    pieces: list[Piece], captions: list[RenderBlock], block: RenderBlock, chapter: str
) -> None:
    if block.failed or block.region is None:
        return
    if block.kind == "figure":
        pieces.append(Piece(block, chapter))
    elif block.kind == "caption":
        captions.append(block)


def _emit(
    group: FigureGroup,
    groups: list[FigureGroup],
    captions: list[RenderBlock],
    loaded: BookLoad,
    run_dir: Path,
    paths: dict[str, str],
    blobs: dict[str, Asset],
    caption_text: dict[str, str],
    suppressed: set[str],
    rows: list[dict[str, object]],
) -> None:
    region = group.primary.region
    if region is None:
        return
    caption = caption_in_crop(group, captions)
    caption_box = from_region(caption.region) if caption is not None and caption.region else None
    box = padded_figure(
        group.union, region.width, region.height, caption_box, _others(group, groups)
    )
    cropped = crop_figure(
        run_dir, group.primary.page_id, loaded.images.get(group.primary.pdf_page), box
    )
    if cropped is None:
        return
    image, variant, recorded = cropped
    _store(group, image, variant, recorded, captions, paths, blobs, caption_text, suppressed, rows)


def _store(
    group: FigureGroup,
    image: Crop,
    variant: str,
    recorded: tuple[int, int, int, int],
    captions: list[RenderBlock],
    paths: dict[str, str],
    blobs: dict[str, Asset],
    caption_text: dict[str, str],
    suppressed: set[str],
    rows: list[dict[str, object]],
) -> None:
    number = group.number or ""
    rel = figure_asset(
        group.pieces[0].chapter,
        group.primary.page_id,
        group.primary.reading_order,
        "figure",
        number,
    )
    paths[group.primary.block_id] = rel
    blobs[rel] = image
    text = _caption_text(group.caption_id, captions)
    if text:
        caption_text[group.primary.block_id] = text
    if group.caption_id:
        suppressed.add(group.caption_id)
    for piece in group.pieces[1:]:
        suppressed.add(piece.block.block_id)
    rows.append(figure_row(group, rel, variant, recorded))


def _caption_text(caption_id: str | None, captions: list[RenderBlock]) -> str:
    if not caption_id:
        return ""
    for block in captions:
        if block.block_id == caption_id:
            return block.text.strip()
    return ""


def _others(group: FigureGroup, groups: list[FigureGroup]) -> list[Box]:
    page = group.primary.pdf_page
    return [item.union for item in groups if item is not group and item.primary.pdf_page == page]


def _rejected(pieces: list[Piece], ornaments: set[str], noise: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for piece in pieces:
        block_id = piece.block.block_id
        if block_id in ornaments:
            rows.append(rejected_row(piece.block, REASON_ORNAMENT))
        elif block_id in noise:
            rows.append(rejected_row(piece.block, REASON_NOISE))
    return rows
