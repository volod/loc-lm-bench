"""Prepare a blind reference draft and keep unmatched or differing readings visible.

Layout regions own the lines that sit inside them. A line with no region stays a block at its
own crop, and a line two regions claim equally keeps both crops on the note.
"""

from dataclasses import asdict
from typing import Any

from llb.prep.book.ledger.reading import Group, Placed, arrange
from llb.prep.book.metrics.geometry import Box, nearest_links
from llb.prep.book.metrics.latex import latex_parses
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.metrics.tables import parse_table_text

_UNOWNED = "No region owns this line."
_UNCLEAR = "Placement is unclear."
_SINGLE = "Only one reading covers this block."


def merge_readings(
    primary: PageSubmission, alternate: PageSubmission
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Place each line in the region that owns it. Unmatched lines stay at their own crop."""
    regions = tuple(sorted(primary.blocks, key=lambda block: block.reading_order))
    lines = tuple(sorted(alternate.blocks, key=lambda block: block.reading_order))
    height = _page_height(regions, lines)
    groups = arrange(
        [_placed(block, index) for index, block in enumerate(regions)],
        [_placed(block, index) for index, block in enumerate(lines)],
        page_height=height,
    )
    blocks: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    for group in groups:
        reading, note = _draft_note(group, regions, lines, len(blocks))
        blocks.append(reading)
        notes.append(note)
    _link_figures(blocks, notes)
    return blocks, [note for note in notes if note["flags"]]


def printed_page(blocks: list[dict[str, Any]]) -> str | None:
    """Only suggest a folio when one labelled numeric block exists."""
    options = [block["text"].strip() for block in blocks if block["kind"] == "page-number"]
    return str(options[0]) if len(options) == 1 and options[0].isdecimal() else None


def _draft_note(
    group: Group,
    regions: tuple[HypBlock, ...],
    lines: tuple[HypBlock, ...],
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    owned = [lines[line] for line in group.lines]
    rivals = [regions[rival] for rival in group.rivals]
    reading, structure_flags = _draft_group(group.region, group.kind, owned, regions, index)
    flags = _group_flags(group.region, group.rivals, owned, regions) + structure_flags
    shown = rivals if group.rivals else owned
    note = {
        "block_id": reading["block_id"],
        "flags": flags,
        "alternate": [_alternate(item) for item in shown] if flags else [],
    }
    return reading, note


def _draft_group(
    region_index: int | None,
    kind_override: str | None,
    owned: list[HypBlock],
    regions: tuple[HypBlock, ...],
    index: int,
) -> tuple[dict[str, Any], list[str]]:
    source, text, kind = _draft_source(region_index, kind_override, owned, regions)
    reading: dict[str, Any] = {
        "block_id": f"b{index + 1:04d}",
        "kind": kind,
        "region": asdict(_box(source)),
        "reading_order": index,
        "text": text,
    }
    return reading, _structure_flags(reading, source, text, kind)


def _draft_source(
    region_index: int | None,
    kind_override: str | None,
    owned: list[HypBlock],
    regions: tuple[HypBlock, ...],
) -> tuple[HypBlock, str, str]:
    if region_index is None:
        source = owned[0]
        return source, _joined(owned), kind_override or source.kind
    source = regions[region_index]
    text = source.text
    if not text and source.kind != "figure":
        text = _joined(owned)
    return source, text, source.kind


def _joined(blocks: list[HypBlock]) -> str:
    return "\n".join(block.text for block in blocks if block.text)


def _structure_flags(reading: dict[str, Any], source: HypBlock, text: str, kind: str) -> list[str]:
    if kind == "formula":
        return _draft_formula(reading, source, text)
    if kind == "table":
        return _draft_table(reading, source, text)
    return []


def _group_flags(
    region_index: int | None,
    rivals: tuple[int, ...],
    owned: list[HypBlock],
    regions: tuple[HypBlock, ...],
) -> list[str]:
    if region_index is None:
        return [_UNCLEAR] if rivals else [_UNOWNED]
    source = regions[region_index]
    if not owned:
        return [_SINGLE]
    flags = []
    joined = " ".join(word for block in owned for word in block.text.split())
    if " ".join(source.text.split()) != joined:
        flags.append("Text differs.")
    if len(owned) != 1 or _box(source) != _box(owned[0]):
        flags.append("Block boundaries differ.")
    if any(block.kind != source.kind for block in owned):
        flags.append("Block kinds differ.")
    if len(owned) == 1 and _structure(source) != _structure(owned[0]):
        flags.append("Technical structure differs.")
    return flags


def _alternate(block: HypBlock) -> dict[str, Any]:
    return {"kind": block.kind, "text": block.text, "region": asdict(_box(block))}


def _placed(block: HypBlock, order: int) -> Placed:
    box = _box(block)
    return Placed(block.kind, order, box.x0, box.y0, box.x1, box.y1, block.text)


def _page_height(regions: tuple[HypBlock, ...], lines: tuple[HypBlock, ...]) -> float:
    boxes = [block.region for block in (*regions, *lines) if block.region is not None]
    return max((box.y1 for box in boxes), default=0.0)


def _box(block: HypBlock) -> Box:
    if block.region is None:
        raise ValueError("pilot drafts require boxed recognition runs")
    return block.region


def _structure(block: HypBlock) -> tuple[object, ...]:
    return (block.latex, block.table, block.equation_number, block.caption_id is not None)


def _draft_formula(reading: dict[str, Any], block: HypBlock, text: str) -> list[str]:
    latex = block.latex or text
    if latex.strip():
        reading["latex"] = latex
        reading["equation_number"] = block.equation_number
        if latex_parses(latex):
            return []
    else:
        reading["kind"] = "unknown"
    return ["Formula needs a verified LaTeX reading."]


def _draft_table(reading: dict[str, Any], block: HypBlock, text: str) -> list[str]:
    rows = block.table or parse_table_text(text)
    if rows:
        reading["table"] = {"rows": [[asdict(cell) for cell in row] for row in rows]}
        return []
    reading["kind"] = "unknown"
    return ["Table needs a verified cell grid."]


def _link_figures(blocks: list[dict[str, Any]], notes: list[dict[str, Any]]) -> None:
    def boxes(kind: str) -> list[tuple[str, Box]]:
        return [(b["block_id"], Box(**b["region"])) for b in blocks if b["kind"] == kind]

    guessed = nearest_links(boxes("figure"), boxes("caption"))
    for block, note in zip(blocks, notes, strict=True):
        if block["kind"] == "figure":
            block["caption_id"] = guessed.get(block["block_id"])
            note["flags"].append("Check the figure bounds and caption association.")
