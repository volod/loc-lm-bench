"""Render one book-ocr run as an Obsidian vault.

The ledger stays the source. This writes the notes, the figure crops, and the
source PDF under the policy the operator chose, then swaps that tree into
place.
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import Region
from llb.prep.book.constants import SOURCE_POLICIES, SOURCE_POLICY_REFERENCE, SOURCE_POLICY_SYMLINK
from llb.prep.book.preprocess.images import encode_png
from llb.prep.book.ranges import page_id
from llb.prep.book.render.constants import (
    OBSIDIAN_VERSION,
    PDF_PAGE_FRAGMENT,
    REFERENCE_FILENAME,
    REFERENCE_REPORT_FILENAME,
    RENDER_MANIFEST_FILENAME,
    SECTION_CAP_FLOOR,
    SECTION_CHAR_CAP,
    SOURCE_DIRNAME,
    SOURCE_PAGES_DIRNAME,
    VAULT_DIRNAME,
)
from llb.prep.book.render.documents import layout_files, render_markdown
from llb.prep.book.render.grammar import load_grammar
from llb.prep.book.render.references import reference_json, reference_line
from llb.prep.book.render.load import load_book
from llb.prep.book.render.names import (
    anchor_map,
    book_slug,
    chapter_slug,
    equation_numbers,
    figure_asset,
    figure_numbers,
    heading_levels,
    parents_of,
    source_filename,
)
from llb.prep.book.render.publish import publish
from llb.prep.book.render.segment import Span, prepare, section_spans
from llb.prep.book.render.types import BookLoad, RenderBlock, RenderContext, VaultFile
from llb.prep.book.review.crops import padded_box
from llb.prep.book.review.constants import CROP_PADDING_PX
from llb.prep.book.source import refuse_source_overlap
from llb.prep.book.structure.io import structure_path

_LOG = logging.getLogger(__name__)
Href = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """What an operator can change without touching the book's structure."""

    output: Path | None = None
    source_policy: str = SOURCE_POLICY_SYMLINK
    page_fallback: bool = False
    section_cap: int = SECTION_CHAR_CAP
    slug: str | None = None
    reference_grammar: Path | None = None


def build_vault(run_dir: Path, options: RenderOptions | None = None) -> Path:
    """Write the vault and return its directory. A second call writes the same bytes."""
    opts = options or RenderOptions()
    _check(opts)
    loaded = load_book(run_dir)
    slug = opts.slug or book_slug(loaded.pdf.name)
    dest = opts.output if opts.output is not None else run_dir / VAULT_DIRNAME / slug
    _refuse(dest, run_dir, loaded.pdf)
    blocks = prepare(loaded.blocks, loaded.structure)
    spans = section_spans(blocks, loaded.structure)
    assets, figures = _figures(spans, loaded, slug)
    ctx = _context(loaded, dest, slug, opts, assets)
    files = layout_files(spans, opts.section_cap, slug, ctx)
    grammar = load_grammar(opts.reference_grammar)
    docs, counts = render_markdown(files, loaded.structure, ctx, grammar)
    blobs = {rel: text.encode("utf-8") for rel, text in docs.items()}
    blobs[f"metadata/{RENDER_MANIFEST_FILENAME}"] = _manifest(loaded, slug, opts, files, run_dir)
    blobs[f"metadata/{REFERENCE_REPORT_FILENAME}"] = reference_json(counts)
    blobs["metadata/document_structure.json"] = structure_path(run_dir).read_bytes()
    blobs.update(figures)
    blobs.update(_crops(blocks, loaded))
    blobs.update(_pages(loaded, opts))
    links = _source(loaded, dest, opts, blobs)
    publish(dest, blobs, links)
    failed = sum(1 for block in blocks if block.failed)
    _LOG.info(
        "[book-render] %s content files, %s failed pages, %s -> %s",
        len(files),
        failed,
        reference_line(counts),
        dest,
    )
    return dest


def _check(opts: RenderOptions) -> None:
    if opts.source_policy not in SOURCE_POLICIES:
        raise ValueError(f"unknown source policy {opts.source_policy}")
    if opts.section_cap < SECTION_CAP_FLOOR:
        raise ValueError(f"section cap must be at least {SECTION_CAP_FLOOR}")
    if opts.slug is not None and opts.slug != book_slug(opts.slug):
        raise ValueError("slug must be lowercase ASCII words and hyphens")


def _refuse(dest: Path, run_dir: Path, pdf: Path) -> None:
    dest_real = dest.resolve()
    run_real = run_dir.resolve()
    if dest_real == run_real or dest_real in run_real.parents:
        raise ValueError(f"refusing to replace the run directory {run_dir}")
    refuse_source_overlap(pdf, dest)


def _context(
    loaded: BookLoad,
    dest: Path,
    slug: str,
    opts: RenderOptions,
    assets: dict[str, str],
) -> RenderContext:
    pdf = _vault_pdf(dest, loaded.pdf, opts.source_policy)
    return RenderContext(
        book=slug,
        fallback=opts.page_fallback,
        anchors=anchor_map(loaded.structure),
        equations=equation_numbers(loaded.structure),
        assets=assets,
        levels=heading_levels(loaded.structure),
        content_href=_href(dest / "content", pdf),
        toc_href=_href(dest, pdf),
        pages_with_images=frozenset(loaded.images),
    )


def _vault_pdf(dest: Path, pdf: Path, policy: str) -> Path:
    if policy == SOURCE_POLICY_REFERENCE:
        return pdf.resolve()
    return dest / SOURCE_DIRNAME / source_filename(pdf.name)


def _href(directory: Path, pdf: Path) -> Href:
    def href(page: int) -> str:
        rel = Path(os.path.relpath(pdf, directory)).as_posix()
        return f"{rel}#{PDF_PAGE_FRAGMENT}={page}"

    return href


def _figures(
    spans: list[Span], loaded: BookLoad, book: str
) -> tuple[dict[str, str], dict[str, bytes]]:
    parents = parents_of(loaded.structure.hierarchy)
    numbers = figure_numbers(loaded.structure)
    paths: dict[str, str] = {}
    blobs: dict[str, bytes] = {}
    for span in spans:
        chapter = chapter_slug(span.node, parents, book)
        for block in span.blocks:
            _figure(block, chapter, numbers, loaded, paths, blobs)
    return paths, blobs


def _figure(
    block: RenderBlock,
    chapter: str,
    numbers: dict[str, str],
    loaded: BookLoad,
    paths: dict[str, str],
    blobs: dict[str, bytes],
) -> None:
    if block.kind != "figure" or block.region is None:
        return
    image = loaded.images.get(block.pdf_page)
    if image is None:
        return
    data = _region_png(image, block.region, 0)
    if data is None:
        return
    rel = figure_asset(
        chapter, block.page_id, block.ordinal, block.kind, numbers.get(block.block_id, "")
    )
    paths[block.block_id] = rel
    blobs[rel] = data


def _crops(blocks: list[RenderBlock], loaded: BookLoad) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    for block in blocks:
        image = loaded.images.get(block.pdf_page)
        if block.region is None or image is None:
            continue
        data = _region_png(image, block.region, CROP_PADDING_PX)
        if data is None:
            continue
        for _start, _end, rel in block.crops:
            blobs.setdefault(rel, data)
    return blobs


def _pages(loaded: BookLoad, opts: RenderOptions) -> dict[str, bytes]:
    if not opts.page_fallback:
        return {}
    blobs: dict[str, bytes] = {}
    width = loaded.config.source.page_count
    for pdf_page, image in sorted(loaded.images.items()):
        pid = page_id(pdf_page, width)
        blobs[f"{SOURCE_DIRNAME}/{SOURCE_PAGES_DIRNAME}/{pid}.png"] = encode_png(image)
    return blobs


def _source(
    loaded: BookLoad, dest: Path, opts: RenderOptions, blobs: dict[str, bytes]
) -> dict[str, Path]:
    name = source_filename(loaded.pdf.name)
    rel = f"{SOURCE_DIRNAME}/{name}"
    if opts.source_policy == "copy":
        blobs[rel] = loaded.pdf.read_bytes()
        return {}
    if opts.source_policy == SOURCE_POLICY_SYMLINK:
        return {rel: loaded.pdf.resolve()}
    blobs[f"{SOURCE_DIRNAME}/{REFERENCE_FILENAME}"] = _reference(dest, loaded.pdf)
    return {}


def _reference(dest: Path, pdf: Path) -> bytes:
    rel = Path(os.path.relpath(pdf.resolve(), dest.resolve())).as_posix()
    return (rel + "\n").encode("ascii", "replace")


def _manifest(
    loaded: BookLoad, slug: str, opts: RenderOptions, files: list[VaultFile], run_dir: Path
) -> bytes:
    payload = {
        "book": slug,
        "content_files": [f"content/{file.stem}.md" for file in files],
        "obsidian_version": OBSIDIAN_VERSION,
        "page_fallback": opts.page_fallback,
        "pdf_page_fragment": PDF_PAGE_FRAGMENT,
        "run_dir": str(run_dir.resolve()),
        "section_char_cap": opts.section_cap,
        "source_policy": opts.source_policy,
        "source_sha256": loaded.config.source.sha256,
    }
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return text.encode("ascii")


def _region_png(image: np.ndarray, region: Region, padding: int) -> bytes | None:
    box = padded_box(region, image.shape, padding)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return encode_png(image[y0:y1, x0:x1])
