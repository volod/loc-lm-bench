"""Figure assets: one padded crop, the printed caption, and no repeated ornament."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookSourceRecord, BookStageSpec
from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, RawReading, Region
from llb.core.contracts.data_prep.book_preprocess import GridTransform
from llb.core.contracts.data_prep.book_resolve import PageResolvedText, ResolvedBlock, ResolvedSpan
from llb.core.contracts.data_prep.book_structure import (
    STRUCTURE_KIND,
    STRUCTURE_SCHEMA_ID,
    STRUCTURE_SCHEMA_VERSION,
    DocumentStructure,
    NumberedItem,
    PrintedPage,
    StructureNode,
)
from llb.prep.book.assets.boxes import Box
from llb.prep.book.assets.boxes import span_limit
from llb.prep.book.assets.constants import (
    CAPTION_GAP_FRAC,
    CAPTION_GAP_MAX_PX,
    CAPTION_GAP_MIN_PX,
    FIGURE_RECORD_FILENAME,
    GROUP_GAP_FRAC,
    GROUP_GAP_MAX_PX,
    GROUP_GAP_MIN_PX,
    REASON_NOISE,
    REASON_ORNAMENT,
)
from llb.prep.book.assets.crop import best_geometric, original_box_to_variant
from llb.prep.book.config import write_config
from llb.prep.book.constants import BLOCK_LEDGER_SCHEMA_ID, CONTRACT_VERSION, RUN_CONFIG_SCHEMA_ID
from llb.prep.book.ledger.io import write_ledger
from llb.prep.book.preprocess.constants import MANIFEST_FILENAME, ORIGINAL, VARIANTS_STAGE
from llb.prep.book.preprocess.geometry import Affine, identity
from llb.prep.book.preprocess.images import read_pixels, write_png
from llb.prep.book.preprocess.records import (
    affine_record,
    kept,
    page_manifest,
    refused,
    write_manifest,
)
from llb.prep.book.ranges import page_id
from llb.prep.book.render.build import RenderOptions, build_vault
from llb.prep.book.resolve.constants import RESOLVED_KIND, RESOLVED_SCHEMA_ID, SCHEMA_VERSION
from llb.prep.book.resolve.io import write_resolved
from llb.prep.book.structure.io import write_structure

_W = 800
_H = 1200


def test_best_geometric_skips_a_refusal_and_prefers_dewarp() -> None:
    deskew = page_manifest(
        "p0001",
        1,
        [
            kept("rotated", "rotated/page.png", affine_record(identity(8, 8)), {}, geometric=True),
            refused("dewarp", "stroke", {}, geometric=True),
            kept("deskew", "deskew/page.png", affine_record(identity(8, 8)), {}, geometric=True),
        ],
    )
    chosen = best_geometric(deskew)
    assert chosen is not None and chosen.name == "deskew"
    dewarp = page_manifest(
        "p0001",
        1,
        [
            kept("deskew", "deskew/page.png", affine_record(identity(8, 8)), {}, geometric=True),
            kept("dewarp", "dewarp/page.png", affine_record(identity(8, 8)), {}, geometric=True),
        ],
    )
    preferred = best_geometric(dewarp)
    assert preferred is not None and preferred.name == "dewarp"


def test_shifted_affine_maps_the_original_box() -> None:
    transform = affine_record(Affine(((1.0, 0.0, -10.0), (0.0, 1.0, -8.0)), 80, 60, 80, 60))
    mapped = original_box_to_variant(transform, Box(20, 20, 40, 40))
    assert mapped is not None
    assert mapped.x0 == pytest.approx(30)
    assert mapped.y0 == pytest.approx(28)
    assert mapped.x1 == pytest.approx(50)
    assert mapped.y1 == pytest.approx(48)


def test_identity_grid_covers_the_original_box() -> None:
    transform = GridTransform(
        kind="grid",
        rows=[0, 59],
        cols=[0, 79],
        src_width=80,
        src_height=60,
        width=80,
        height=60,
        map_x=[[0.0, 79.0], [0.0, 79.0]],
        map_y=[[0.0, 0.0], [59.0, 59.0]],
    )
    mapped = original_box_to_variant(transform, Box(16, 16, 40, 40))
    assert mapped is not None
    assert mapped.x0 <= 16 and mapped.y0 <= 16
    assert mapped.x1 >= 40 and mapped.y1 >= 40


def test_split_figure_is_one_asset_and_names_follow_reading_order(tmp_path: Path) -> None:
    gap = span_limit(_W, _H, GROUP_GAP_FRAC, GROUP_GAP_MIN_PX, GROUP_GAP_MAX_PX)
    upper = _block(1, 9, 1, "figure", "", _box(100, 200, 420, 400))
    lower = _block(1, 3, 2, "figure", "", _box(100, 400 + gap / 2, 420, 560))
    caption = _block(1, 4, 3, "caption", "Figure 2-1 The plot", _box(100, 580, 420, 620))
    other = _block(1, 2, 4, "figure", "", _box(100, 800, 420, 980))
    other_caption = _block(1, 5, 5, "caption", "Figure 2-2 The other", _box(100, 990, 420, 1030))
    items = [
        _item("2-1", 1, caption.block_id),
        _item("2-2", 1, other_caption.block_id),
    ]
    pages = {1: [upper, lower, caption, other, other_caption]}
    run = _write(tmp_path, pages, items)
    first = build_vault(run, RenderOptions(output=tmp_path / "a"))
    second = build_vault(run, RenderOptions(output=tmp_path / "b"))
    record = _record(first)
    again = _record(second)
    assert record == again
    assert [row["asset"] for row in record["figures"]] == sorted(
        row["asset"] for row in record["figures"]
    )
    assert len(record["figures"]) == 2
    split, separate = record["figures"]
    assert split["asset"].endswith("p0001_0001_figure_2-1.png")
    assert len(split["regions"]) == 2
    assert {region["block_id"] for region in split["regions"]} == {upper.block_id, lower.block_id}
    assert split["caption_block_id"] == caption.block_id
    assert separate["asset"].endswith("p0001_0004_figure_2-2.png")
    assert separate["caption_block_id"] == other_caption.block_id
    body = _body(first)
    # The citation pass links "Figure 2-1", so the caption remainder is what stays whole.
    assert body.count("The plot") == 1
    assert body.count("The other") == 1
    assert body.count("![[") == 2
    assert "[p0001](" in body
    _same_bytes(first, second)


def test_repeated_ornament_never_becomes_a_figure(tmp_path: Path) -> None:
    badge_box = _box(24, 40, 64, 80)
    drawing = _block(1, 1, 1, "figure", "", _box(150, 250, 560, 700))
    pages = {
        1: [_block(1, 0, 0, "figure", "LOGO", badge_box), drawing],
        2: [_block(2, 0, 0, "figure", "LOGO", badge_box)],
    }
    vault = _render(tmp_path, pages, [])
    record = _record(vault)
    assert [row["block_id"] for row in record["figures"]] == [drawing.block_id]
    assert {row["reason"] for row in record["rejected"]} == {REASON_ORNAMENT}
    assert len(record["rejected"]) == 2
    body = _body(vault)
    assert body.count("![[") == 1
    assert "> [ornament] LOGO" in body


def test_isolated_noise_is_not_a_figure(tmp_path: Path) -> None:
    speck = _block(1, 0, 0, "figure", "SPECK", _box(300, 300, 316, 316))
    vault = _render(tmp_path, {1: [speck]}, [])
    record = _record(vault)
    assert record["figures"] == []
    assert record["rejected"] == [
        {
            "block_id": speck.block_id,
            "page_id": "p0001",
            "pdf_page": 1,
            "reason": REASON_NOISE,
        }
    ]
    assert "![[" not in _body(vault)
    assert "> [ornament] SPECK" in _body(vault)


def test_wrong_caption_cannot_replace_the_printed_number(tmp_path: Path) -> None:
    figure = _block(1, 0, 0, "figure", "", _box(80, 200, 500, 500))
    near = _block(1, 1, 1, "caption", "Figure 9-9 Other", _box(80, 508, 500, 548))
    limit = span_limit(_W, _H, CAPTION_GAP_FRAC, CAPTION_GAP_MIN_PX, CAPTION_GAP_MAX_PX)
    far = _block(
        1,
        2,
        2,
        "caption",
        "Figure 4-1 The orbit",
        _box(80, 500 + limit + 40, 500, 500 + limit + 80),
    )
    vault = _render(tmp_path, {1: [figure, near, far]}, [_item("4-1", 1, figure.block_id)])
    record = _record(vault)
    assert len(record["figures"]) == 1
    row = record["figures"][0]
    assert row["number"] == "4-1"
    assert row["caption_block_id"] == far.block_id
    assert row["asset"].endswith("_figure_4-1.png")
    body = _body(vault)
    orbit = body.index("The orbit")
    other = body.index("Figure 9-9 Other")
    assert orbit < other
    assert "The orbit" in body[body.index("![[") : other]
    assert "Figure 9-9" not in body[body.index("![[") : other]


def test_ambiguous_captions_stay_unattached(tmp_path: Path) -> None:
    figure = _block(1, 0, 0, "figure", "", _box(80, 200, 500, 500))
    alpha = _block(1, 1, 1, "caption", "Figure 4-1 Alpha", _box(80, 520, 500, 560))
    beta = _block(1, 2, 2, "caption", "Figure 4-1 Beta", _box(80, 580, 500, 620))
    items = [_item("4-1", 1, alpha.block_id), _item("4-1", 1, beta.block_id)]
    vault = _render(tmp_path, {1: [figure, alpha, beta]}, items)
    row = _record(vault)["figures"][0]
    assert row["number"] is None
    assert row["caption_block_id"] is None
    assert row["asset"].endswith("_figure.png")
    body = _body(vault)
    assert "Figure 4-1 Alpha" in body and "Figure 4-1 Beta" in body
    plain = _block(1, 0, 0, "figure", "", _box(80, 200, 500, 400))
    left = _block(1, 1, 1, "caption", "Alpha note", _box(80, 430, 400, 460))
    right = _block(1, 2, 2, "caption", "Beta note", _box(420, 434, 760, 464))
    other = _render(tmp_path / "near", {1: [plain, left, right]}, [])
    near = _record(other)["figures"][0]
    assert near["number"] is None and near["caption_block_id"] is None


def test_footer_number_is_not_a_slug(tmp_path: Path) -> None:
    figure = _block(1, 0, 0, "figure", "", _box(100, 200, 500, 600))
    item = _item("8-1", 1, figure.block_id)
    item = item.model_copy(update={"evidence": "page-footer"})
    vault = _render(tmp_path, {1: [figure]}, [item])
    row = _record(vault)["figures"][0]
    assert row["number"] is None
    assert "8-1" not in row["asset"]
    assert row["asset"].endswith("_figure.png")


def test_printed_caption_counts_when_inventory_says_footer(tmp_path: Path) -> None:
    figure = _block(1, 0, 0, "figure", "", _box(100, 200, 500, 560))
    caption = _block(1, 1, 1, "caption", "Fig. 2-10 The bus", _box(80, 564, 540, 610))
    item = _item("2-10", 1, caption.block_id).model_copy(update={"evidence": "page-footer"})
    vault = _render(tmp_path, {1: [figure, caption]}, [item])
    row = _record(vault)["figures"][0]
    assert row["number"] == "2-10"
    assert row["caption_block_id"] == caption.block_id
    assert row["asset"].endswith("_figure_2-10.png")
    assert _body(vault).count("The bus") == 1


def test_table_caption_is_not_a_figure_caption(tmp_path: Path) -> None:
    figure = _block(1, 0, 0, "figure", "", _box(80, 160, 140, 230))
    caption = _block(1, 1, 1, "caption", "Table 4-1 The costs", _box(160, 150, 760, 220))
    vault = _render(tmp_path, {1: [figure, caption]}, [])
    row = _record(vault)["figures"][0]
    assert row["number"] is None
    assert row["caption_block_id"] is None
    assert "Table 4-1 The costs" in _body(vault)


def test_crop_uses_the_dewarp_variant_and_records_the_original_region(tmp_path: Path) -> None:
    figure = _block(1, 0, 0, "figure", "", _box(100, 200, 400, 500))
    run = _write(tmp_path, {1: [figure]}, [])
    _variants(run)
    vault = build_vault(run, RenderOptions(output=tmp_path / "vault"))
    row = _record(vault)["figures"][0]
    assert row["variant"] == "dewarp"
    assert row["regions"][0]["x0"] == 100
    assert row["regions"][0]["y0"] == 200
    crop = read_pixels(vault / row["asset"])
    assert 40 in set(np.unique(crop).tolist())
    assert 7 not in set(np.unique(crop).tolist())


def _render(root: Path, pages: dict[int, list[Block]], items: list[NumberedItem]) -> Path:
    run = _write(root, pages, items)
    return build_vault(run, RenderOptions(output=root / "vault"))


def _write(root: Path, pages: dict[int, list[Block]], items: list[NumberedItem]) -> Path:
    run = root / "run"
    pdf = run / "source" / "book.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n%figures\n")
    numbers = sorted(pages)
    _config(run, pdf, numbers)
    for number, blocks in pages.items():
        _save(run, max(numbers), blocks)
    chapter = _node("book/chapter-1", "chapter", "1", "Flight", numbers[0], numbers[-1])
    root_node = _node("book", "book", "", "Book", numbers[0], numbers[-1], [chapter])
    write_structure(
        run / "metadata" / "document_structure.json",
        DocumentStructure(
            schema_id=STRUCTURE_SCHEMA_ID,
            schema_version=STRUCTURE_SCHEMA_VERSION,
            kind=STRUCTURE_KIND,
            source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
            pages=[_printed(number) for number in numbers],
            hierarchy=root_node,
            inventory=items,
        ),
    )
    return run


def _variants(run: Path) -> None:
    directory = run / "pages" / "p0001" / VARIANTS_STAGE
    original = np.full((_H, _W), 200, dtype=np.uint8)
    deskew = np.full((_H, _W), 7, dtype=np.uint8)
    dewarp = np.full((_H, _W), 200, dtype=np.uint8)
    dewarp[208:508, 110:410] = 40
    write_png(run / "pages" / "p0001" / "render" / "page.png", original)
    write_png(directory / "deskew" / "page.png", deskew)
    write_png(directory / "dewarp" / "page.png", dewarp)
    shift = Affine(((1.0, 0.0, -10.0), (0.0, 1.0, -8.0)), _W, _H, _W, _H)
    write_manifest(
        directory / MANIFEST_FILENAME,
        page_manifest(
            "p0001",
            1,
            [
                kept(ORIGINAL, None, affine_record(identity(_W, _H)), {}, geometric=False),
                kept(
                    "deskew", "deskew/page.png", affine_record(identity(_W, _H)), {}, geometric=True
                ),
                kept("dewarp", "dewarp/page.png", affine_record(shift), {}, geometric=True),
            ],
        ),
    )


def _config(run: Path, pdf: Path, pages: list[int]) -> None:
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    write_config(
        run,
        BookOcrRunConfig(
            schema_id=RUN_CONFIG_SCHEMA_ID,
            schema_version=CONTRACT_VERSION,
            kind="book-ocr-run-config",
            run_id="figure-assets",
            created_at="2026-01-01T00:00:00+00:00",
            source=BookSourceRecord(
                original_path=str(pdf.resolve()),
                sha256=digest,
                size_bytes=pdf.stat().st_size,
                page_count=max(pages),
                policy="copy",
                stored_path="source/book.pdf",
            ),
            pages_spec="1-",
            pages=pages,
            batch_pages=25,
            max_attempts=2,
            stages=[BookStageSpec(name="render")],
        ),
    )


def _save(run: Path, page_count: int, blocks: list[Block]) -> None:
    pdf = blocks[0].pdf_page
    pid = page_id(pdf, page_count)
    base = run / "pages" / pid
    write_ledger(
        base / "blocks.json",
        PageBlockLedger(
            schema_id=BLOCK_LEDGER_SCHEMA_ID,
            schema_version=CONTRACT_VERSION,
            kind="book-ocr-block-ledger",
            page_id=pid,
            pdf_page=pdf,
            blocks=blocks,
        ),
    )
    write_resolved(
        base / "resolved.json",
        PageResolvedText(
            schema_id=RESOLVED_SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            kind=RESOLVED_KIND,
            page_id=pid,
            pdf_page=pdf,
            page_class="",
            primary_stage="ocr",
            primary_engine="rapidocr",
            primary_variant="original",
            blocks=[_resolved(block) for block in blocks],
        ),
    )
    write_png(base / "render" / "page.png", np.full((_H, _W), 210, dtype=np.uint8))


def _block(pdf: int, ordinal: int, reading: int, kind: str, text: str, region: Region) -> Block:
    return Block(
        block_id=f"p{pdf:04d}-b{ordinal:04d}",
        pdf_page=pdf,
        ordinal=ordinal,
        kind=kind,  # type: ignore[arg-type]
        region=region,
        reading_order=reading,
        readings=[
            RawReading(
                engine="rapidocr", model=None, variant="original", text=text, confidence=0.99
            )
        ],
    )


def _box(x0: float, y0: float, x1: float, y1: float) -> Region:
    return Region(x0=x0, y0=y0, x1=x1, y1=y1, width=_W, height=_H)


def _resolved(block: Block) -> ResolvedBlock:
    text = block.readings[0].text
    return ResolvedBlock(
        block_id=block.block_id,
        anchor="a" * 64,
        engine="rapidocr",
        variant="original",
        fallback=False,
        primary_text=text,
        text=text,
        spans=[ResolvedSpan(provenance="primary", text=text, start=0, end=len(text), adopted=True)],
    )


def _item(number: str, pdf: int, block_id: str) -> NumberedItem:
    return NumberedItem(
        kind="figure",
        number=number,
        pdf_page=pdf,
        printed=str(pdf),
        block_id=block_id,
        evidence="caption",
    )


def _node(
    node_id: str,
    level: str,
    number: str,
    title: str,
    start: int,
    end: int,
    children: list[StructureNode] | None = None,
) -> StructureNode:
    return StructureNode(
        node_id=node_id,
        level=level,  # type: ignore[arg-type]
        number=number,
        title=title,
        pdf_start=start,
        pdf_end=end,
        printed_start=str(start),
        printed_end=str(end),
        block_id=None,
        evidence=["numbered-heading"] if level != "book" else ["root"],
        fallback=False,
        children=children or [],
    )


def _printed(pdf: int) -> PrintedPage:
    return PrintedPage(
        pdf_page=pdf,
        page_id=f"p{pdf:04d}",
        printed=str(pdf),
        scheme="arabic",
        status="observed",
        block_id=None,
        fallback=False,
    )


def _record(vault: Path) -> dict[str, list[dict[str, object]]]:
    loaded = json.loads((vault / "metadata" / FIGURE_RECORD_FILENAME).read_text(encoding="ascii"))
    assert isinstance(loaded, dict)
    return loaded


def _body(vault: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((vault / "content").glob("*.md"))
    )


def _same_bytes(left: Path, right: Path) -> None:
    for path in sorted(left.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        other = right / path.relative_to(left)
        assert other.read_bytes() == path.read_bytes()
