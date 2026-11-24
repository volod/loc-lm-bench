"""The reading vault: navigation, anchors, splits, and a byte-identical re-render."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookSourceRecord, BookStageSpec
from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, RawReading, Region
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
from llb.prep.book.config import write_config
from llb.prep.book.constants import BLOCK_LEDGER_SCHEMA_ID, CONTRACT_VERSION, RUN_CONFIG_SCHEMA_ID
from llb.prep.book.ledger.io import write_ledger
from llb.prep.book.preprocess.images import read_pixels, write_png
from llb.prep.book.ranges import page_id
from llb.prep.book.render.build import RenderOptions, build_vault
from llb.prep.book.render.constants import OBSIDIAN_VERSION, PDF_PAGE_FRAGMENT
from llb.prep.book.render.segment import Span, pack_span
from llb.prep.book.render.types import RenderBlock
from llb.prep.book.resolve.constants import RESOLVED_KIND, RESOLVED_SCHEMA_ID, SCHEMA_VERSION
from llb.prep.book.resolve.io import write_resolved
from llb.prep.book.structure.io import write_structure

_W = 800
_H = 1200
_ANCHOR = re.compile(r"\^(?:ch|sec|sub|part|fig|tbl|eq)-[A-Za-z0-9-]+")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_EMBED = re.compile(r"!\[\[([^\]|#]+)")
RUNNER = CliRunner()


def _region(y: float) -> Region:
    return Region(x0=80, y0=y, x1=520, y1=y + 28, width=_W, height=_H)


def _block(pdf: int, ordinal: int, kind: str, text: str, y: float) -> Block:
    return Block(
        block_id=f"p{pdf:04d}-b{ordinal:04d}",
        pdf_page=pdf,
        ordinal=ordinal,
        kind=kind,  # type: ignore[arg-type]
        region=_region(y),
        reading_order=ordinal,
        readings=[
            RawReading(
                engine="rapidocr", model=None, variant="original", text=text, confidence=0.99
            )
        ],
    )


def _adopted(block: Block, text: str | None = None) -> ResolvedBlock:
    body = block.readings[0].text if text is None else text
    return ResolvedBlock(
        block_id=block.block_id,
        anchor="a" * 64,
        engine="rapidocr",
        variant="original",
        fallback=False,
        primary_text=body,
        text=body,
        spans=[ResolvedSpan(provenance="primary", text=body, start=0, end=len(body), adopted=True)],
    )


def _unresolved(block: Block) -> ResolvedBlock:
    primary = "ALPHA 0 BETA"
    start = primary.index("0")
    adopted = "ALPHA  BETA"
    spans = [
        ResolvedSpan(provenance="primary", text="ALPHA ", start=0, end=start, adopted=True),
        ResolvedSpan(
            provenance="unresolved",
            text="0",
            start=start,
            end=start + 1,
            adopted=False,
            classes=["digit"],
        ),
        ResolvedSpan(
            provenance="primary", text=" BETA", start=start + 1, end=len(primary), adopted=True
        ),
    ]
    return ResolvedBlock(
        block_id=block.block_id,
        anchor="b" * 64,
        engine="rapidocr",
        variant="original",
        fallback=False,
        primary_text=primary,
        text=adopted,
        spans=spans,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(run: Path, pdf: Path, pages: list[int]) -> None:
    write_config(
        run,
        BookOcrRunConfig(
            schema_id=RUN_CONFIG_SCHEMA_ID,
            schema_version=CONTRACT_VERSION,
            kind="book-ocr-run-config",
            run_id="vault-test",
            created_at="2026-01-01T00:00:00+00:00",
            source=BookSourceRecord(
                original_path=str(pdf.resolve()),
                sha256=_sha(pdf),
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


def _save_page(
    run: Path,
    page_count: int,
    blocks: list[Block],
    resolved: list[ResolvedBlock] | None,
    image: bool,
) -> None:
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
    if resolved is not None:
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
                blocks=resolved,
            ),
        )
    if image:
        write_png(base / "render" / "page.png", np.full((_H, _W), 210, dtype=np.uint8))


def _node(
    node_id: str,
    level: str,
    number: str,
    title: str,
    start: int,
    end: int,
    block_id: str | None,
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
        block_id=block_id,
        evidence=["numbered-heading"] if block_id else ["root"],
        fallback=False,
        children=children or [],
    )


def _printed(pdf: int, block: str | None = None) -> PrintedPage:
    return PrintedPage(
        pdf_page=pdf,
        page_id=f"p{pdf:04d}",
        printed=str(pdf),
        scheme="arabic",
        status="observed",
        block_id=block,
        fallback=False,
    )


def _item(kind: str, number: str, pdf: int, block_id: str) -> NumberedItem:
    return NumberedItem(
        kind=kind,  # type: ignore[arg-type]
        number=number,
        pdf_page=pdf,
        printed=str(pdf),
        block_id=block_id,
        evidence="caption",
    )


def _write_structure(
    run: Path,
    sha: str,
    pages: list[PrintedPage],
    root: StructureNode,
    items: list[NumberedItem],
) -> None:
    write_structure(
        run / "metadata" / "document_structure.json",
        DocumentStructure(
            schema_id=STRUCTURE_SCHEMA_ID,
            schema_version=STRUCTURE_SCHEMA_VERSION,
            kind=STRUCTURE_KIND,
            source_sha256=sha,
            pages=pages,
            hierarchy=root,
            inventory=items,
        ),
    )


def _sample(tmp: Path, shift: int = 0) -> Path:
    """Three pages: a chapter, a section, and a page whose resolve never finished."""
    run = tmp / f"run-{shift}"
    pdf = run / "source" / "book.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n%vault\n")
    pages = [1, 2, 3]
    _config(run, pdf, pages)

    def bid(pdf_page: int, ordinal: int) -> str:
        return f"p{pdf_page:04d}-b{ordinal + shift:04d}"

    def block(pdf_page: int, ordinal: int, kind: str, text: str) -> Block:
        made = _block(pdf_page, ordinal + shift, kind, text, y=40 + ordinal * 80)
        return made.model_copy(update={"block_id": bid(pdf_page, ordinal)})

    page1 = [
        block(1, 0, "page-number", "PN-PAGE-ONE"),
        block(1, 1, "heading", "TOK-HEAD-CH"),
        block(1, 2, "body", "TOK-BODY-CH"),
        block(1, 3, "formula", "TOK-FORMULA"),
        block(1, 4, "table", "| a | b |\n| --- | --- |\n| 1 | 2 |"),
        block(1, 5, "figure", "TOK-FIGURE"),
        block(1, 6, "caption", "TOK-CAPTION"),
    ]
    raw = block(2, 1, "body", "ALPHA O BETA")
    page2 = [block(2, 0, "heading", "TOK-HEAD-SEC"), raw, block(2, 2, "body", "TOK-BODY-SEC")]
    failed = [block(3, 0, "body", "RAW-FAILED-PAGE")]
    _save_page(run, 3, page1, [_adopted(item) for item in page1], True)
    _save_page(run, 3, page2, [_adopted(page2[0]), _unresolved(raw), _adopted(page2[2])], True)
    _save_page(run, 3, failed, None, False)
    chapter = _node("book/chapter-1", "chapter", "1", "Introduction", 1, 3, bid(1, 1))
    section = _node("book/chapter-1/section-1-1", "section", "1.1", "Purpose", 2, 3, bid(2, 0))
    chapter = chapter.model_copy(update={"children": [section]})
    root = _node("book", "book", "", "Book", 1, 3, None, [chapter])
    _write_structure(
        run,
        _sha(pdf),
        [_printed(1, bid(1, 0)), _printed(2), _printed(3)],
        root,
        [
            _item("figure", "1-1", 1, bid(1, 5)),
            _item("table", "2-1", 1, bid(1, 4)),
            _item("equation", "1-1", 1, bid(1, 3)),
        ],
    )
    return run


# Cap 200 is the floor the command accepts. Lengths are chosen so a naive cut at that
# budget lands inside the formula and inside the figure-caption pair.
_CAP = 200
_A = "A" * 180
_F = "F" * 50
_B = "B" * 140
_G = "G" * 30
_C = "C" * 40
_T = "T" * 250
_WORDS = ("WORD " * 50).strip()


def _oversized(tmp: Path) -> Path:
    run = tmp / "wide"
    pdf = run / "source" / "book.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n%wide\n")
    _config(run, pdf, [1])
    blocks = [
        _block(1, 0, "body", _A, 40),
        _block(1, 1, "formula", _F, 140),
        _block(1, 2, "body", _B, 240),
        _block(1, 3, "figure", _G, 340),
        _block(1, 4, "caption", _C, 440),
        _block(1, 5, "body", "D" * 30, 540),
        _block(1, 6, "table", _T, 640),
        _block(1, 7, "body", _WORDS, 740),
    ]
    _save_page(run, 1, blocks, [_adopted(item) for item in blocks], True)
    root = _node("book", "book", "", "Book", 1, 1, None)
    root = root.model_copy(
        update={"children": [_node("book/chapter-1", "chapter", "1", "Wide", 1, 1, None)]}
    )
    _write_structure(run, _sha(pdf), [_printed(1)], root, [])
    return run


def _vault(run: Path, dest: Path, **kwargs: object) -> Path:
    options = RenderOptions(output=dest, **kwargs)  # type: ignore[arg-type]
    return build_vault(run, options)


def _markdown(vault: Path) -> list[Path]:
    return sorted((vault / "content").glob("*.md"))


def _text(vault: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _markdown(vault))


def _front(path: Path) -> dict[str, object]:
    matter = path.read_text(encoding="utf-8").split("---", 2)[1]
    loaded = yaml.safe_load(matter)
    assert isinstance(loaded, dict)
    return loaded


def _snapshot(root: Path) -> dict[str, object]:
    found: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            found[rel] = ("link", os.readlink(path))
        elif path.is_file():
            found[rel] = ("file", path.read_bytes())
    return found


def _assert_links(vault: Path) -> None:
    for path in vault.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for href in _LINK.findall(text):
            target = (path.parent / href.split("#", 1)[0]).resolve()
            assert target.is_file(), href
        for embed in _EMBED.findall(text):
            assert (vault / embed).is_file(), embed


def test_vault_renders_every_block_once_in_order(tmp_path: Path) -> None:
    run = _sample(tmp_path)
    vault = _vault(run, tmp_path / "vault")
    _assert_layout(vault)
    body = _text(vault)
    _assert_reading(body)
    _assert_anchors(body)
    _assert_notes(vault)
    _assert_links(vault)
    _assert_manifest(vault)


def _assert_layout(vault: Path) -> None:
    assert (vault / "00_Table_of_Contents.md").is_file()
    for name in ("content", "assets", "review", "metadata", "source"):
        assert (vault / name).is_dir()


def _assert_reading(body: str) -> None:
    tokens = (
        "TOK-HEAD-CH",
        "TOK-BODY-CH",
        "TOK-FORMULA",
        "TOK-FIGURE",
        "TOK-CAPTION",
        "TOK-HEAD-SEC",
        "ALPHA",
        "TOK-BODY-SEC",
    )
    positions = [body.index(token) for token in tokens]
    assert positions == sorted(positions)
    for token in tokens:
        assert body.count(token) == 1
    assert "RAW-FAILED-PAGE" not in body
    assert "ALPHA O BETA" not in body
    # A non-blank unresolved span is the review queue's low-confidence marker.
    assert "[!low-confidence]" in body
    assert "6-7" in body
    assert "[!page-failed]" in body
    assert "#page=3" in body
    assert r"\tag{1-1}" in body
    assert "| a | b |" in body
    assert "![[assets/" in body
    assert "_figure_1-1.png" in body


def _assert_anchors(body: str) -> None:
    for name in (
        "^ch-1",
        "^sec-1-1",
        "^fig-1-1",
        "^tbl-2-1",
        "^eq-1-1",
        "^p0001",
        "^p0002",
        "^p0003",
    ):
        assert name in body


def _assert_notes(vault: Path) -> None:
    notes = _markdown(vault)
    assert len(notes) == 2
    for note in notes:
        _assert_note(note)
    second = notes[1].read_text(encoding="utf-8")
    assert "Previous" not in notes[0].read_text(encoding="utf-8")
    assert notes[0].name in second
    assert "Next" not in second


def _assert_note(note: Path) -> None:
    text = note.read_text(encoding="utf-8")
    nav = [line for line in text.splitlines() if line.startswith("[Contents]")]
    assert len(nav) == 2
    assert nav[0] == nav[1]
    assert f"#{PDF_PAGE_FRAGMENT}=" in nav[0]
    matter = _front(note)
    assert set(matter) == {
        "book",
        "node_id",
        "pdf_start",
        "pdf_end",
        "printed_start",
        "printed_end",
        "review_count",
    }
    assert isinstance(matter["review_count"], int)


def _assert_manifest(vault: Path) -> None:
    manifest = json.loads((vault / "metadata" / "render.json").read_text(encoding="utf-8"))
    assert manifest["obsidian_version"] == OBSIDIAN_VERSION
    assert manifest["pdf_page_fragment"] == PDF_PAGE_FRAGMENT


def test_rerender_is_byte_identical_and_drops_stale_files(tmp_path: Path) -> None:
    run = _sample(tmp_path)
    vault = _vault(run, tmp_path / "vault")
    first = _snapshot(vault)
    stale = vault / "content" / "stale.md"
    stale.write_text("stale\n", encoding="utf-8")
    _vault(run, vault)
    assert not stale.exists()
    assert _snapshot(vault) == first


def test_source_policy_and_page_fallback_keep_file_names(tmp_path: Path) -> None:
    run = _sample(tmp_path)
    names = None
    for policy in ("symlink", "copy", "reference"):
        vault = _vault(run, tmp_path / policy, source_policy=policy)
        stems = [path.name for path in _markdown(vault)]
        names = stems if names is None else names
        assert stems == names
        _assert_links(vault)
        stored = vault / "source" / "book.pdf"
        if policy == "symlink":
            assert stored.is_symlink()
        elif policy == "copy":
            assert stored.is_file() and not stored.is_symlink()
            assert stored.read_bytes() == (run / "source" / "book.pdf").read_bytes()
        else:
            assert not stored.exists()
            assert (vault / "source" / "reference.txt").is_file()
    plain = _vault(run, tmp_path / "plain", page_fallback=False)
    images = _vault(run, tmp_path / "images", page_fallback=True)
    assert [path.name for path in _markdown(plain)] == [path.name for path in _markdown(images)]
    assert list((images / "source" / "pages").glob("*.png"))
    assert not (plain / "source" / "pages").exists()
    _assert_links(images)


def test_vault_preserves_color_channels_in_crops_and_fallback(tmp_path: Path) -> None:
    run = _sample(tmp_path)
    color = np.full((_H, _W, 3), (11, 71, 201), dtype=np.uint8)
    source = run / "pages/p0001/render/page.png"
    write_png(source, color)
    write_png(run / "pages/p0002/render/page.png", color)
    vault = _vault(run, tmp_path / "color", page_fallback=True)
    assert (vault / "source/pages/p0001.png").read_bytes() == source.read_bytes()
    for directory in ("assets", "review"):
        crop = read_pixels(next((vault / directory).rglob("*.png")))
        assert crop.shape[2] == 3
        assert np.all(crop == (11, 71, 201))


def test_source_replacement_refuses_publication(tmp_path: Path) -> None:
    run = _sample(tmp_path)
    vault = _vault(run, tmp_path / "book", source_policy="copy")
    original = _snapshot(vault)
    (run / "source/book.pdf").write_bytes(b"another edition")
    with pytest.raises(ValueError, match="sha256"):
        _vault(run, vault, source_policy="copy")
    assert _snapshot(vault) == original


def test_oversized_section_splits_only_at_permitted_boundaries(tmp_path: Path) -> None:
    run = _oversized(tmp_path)
    vault = _vault(run, tmp_path / "vault", section_cap=_CAP)
    files = [(path.name, path.read_text(encoding="utf-8")) for path in _markdown(vault)]
    formula = [text for _name, text in files if _F in text]
    assert len(formula) == 1 and _A not in formula[0]
    pair = [text for _name, text in files if _G in text]
    assert len(pair) == 1 and _C in pair[0] and _B not in pair[0]
    table = [text for _name, text in files if _T in text]
    assert len(table) == 1
    assert all(_WORDS not in text for _name, text in files)
    pieces = [
        line
        for _name, text in files
        for line in text.splitlines()
        if line.startswith("WORD ") or line == "WORD"
    ]
    assert " ".join(pieces) == _WORDS
    assert len(_markdown(vault)) > 1


def test_shifted_ordinals_keep_numbered_anchors_and_file_names(tmp_path: Path) -> None:
    first = _vault(_sample(tmp_path, 0), tmp_path / "a")
    second = _vault(_sample(tmp_path, 7), tmp_path / "b")
    assert [path.name for path in _markdown(first)] == [path.name for path in _markdown(second)]
    assert _ANCHOR.findall(_text(first)) == _ANCHOR.findall(_text(second))
    assert (
        "^fig-1-1" in _text(second) and "b0005" not in _text(second).split("^fig-1-1", 1)[0][-20:]
    )


def _line(text: str, x: float, y: float) -> RenderBlock:
    return RenderBlock(
        pdf_page=1,
        page_id="p0001",
        ordinal=0,
        reading_order=0,
        kind="body",
        role="body",
        block_id="",
        digest="",
        text=text,
        region=Region(x0=x, y0=y, x1=x + 120, y1=y + 28, width=800, height=1200),
        resolved=None,
    )


def test_paragraph_joins_a_column_tail_and_leaves_the_footnote() -> None:
    blocks = [
        _line("tech-", 80, 100),
        _line("nology matters.", 80, 124),
        _line("The cost is neces-", 80, 200),
        _line("*See the note.", 80, 260),
        _line("Table 1-0, Fig. 1-1, Eq. 1-0", 80, 340),
        _line("sary and appropriate today.", 500, 200),
    ]
    node = _node("book/chapter-1", "chapter", "1", "Wide", 1, 1, None)
    parts = pack_span(Span(node, 1, blocks), 12000, lambda _page: "book.pdf")
    units = [unit.text for part in parts for unit in part]
    assert units == [
        "technology matters.",
        "The cost is necessary",
        "*See the note.",
        "Table 1-0, Fig. 1-1, Eq. 1-0",
        "and appropriate today.",
    ]


def test_render_command_refuses_a_run_without_structure(tmp_path: Path) -> None:
    run = _sample(tmp_path)
    rendered = RUNNER.invoke(
        app, ["book-ocr", "render", str(run), "--output", str(tmp_path / "out")]
    )
    assert rendered.exit_code == 0
    assert "vault ->" in rendered.stdout
    (run / "metadata" / "document_structure.json").unlink()
    refused = RUNNER.invoke(app, ["book-ocr", "render", str(run)])
    assert refused.exit_code == 2
    assert "document structure" in refused.stderr
