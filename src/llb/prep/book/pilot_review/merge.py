"""Prepare a blind reference draft and keep unmatched or differing readings visible."""

from dataclasses import asdict
from typing import Any

from llb.prep.book.metrics.geometry import Box, intersection_over_union, nearest_links
from llb.prep.book.metrics.latex import latex_parses
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.metrics.tables import parse_table_text


def _overlap(left: Box, right: Box) -> float:
    """Share of the alternate block covered by a layout block (often a paragraph)."""
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return width * height / ((right.x1 - right.x0) * (right.y1 - right.y0))


def _group(
    primary: tuple[HypBlock, ...], alternate: tuple[HypBlock, ...]
) -> tuple[dict[str, list[HypBlock]], list[HypBlock]]:
    matches: dict[str, list[HypBlock]] = {}
    missing = []
    for block in alternate:
        region = _box(block)
        candidates = [
            (intersection_over_union(_box(other), region), overlap, other.block_id)
            for other in primary
            if (overlap := _overlap(_box(other), region)) >= 0.5
        ]
        if not candidates:
            missing.append(block)
            continue
        _, _, key = max(candidates)
        matches.setdefault(key, []).append(block)
    return matches, missing


def _box(block: HypBlock) -> Box:
    if block.region is None:
        raise ValueError("pilot drafts require boxed recognition runs")
    return block.region


def _flags(
    primary: HypBlock, alternate: list[HypBlock], previous_rank: int, position: int
) -> list[str]:
    if not alternate:
        return ["Only one reading covers this block."]
    flags = []
    if " ".join(primary.text.split()) != " ".join(
        word for block in alternate for word in block.text.split()
    ):
        flags.append("Text differs.")
    if len(alternate) != 1 or _box(primary) != _box(alternate[0]):
        flags.append("Block boundaries differ.")
    if any(block.kind != primary.kind for block in alternate):
        flags.append("Block kinds differ.")
    if len(alternate) == 1 and _structure(primary) != _structure(alternate[0]):
        flags.append("Technical structure differs.")
    if position < previous_rank:
        flags.append("Reading order differs.")
    return flags


def _structure(block: HypBlock) -> tuple[object, ...]:
    return (block.latex, block.table, block.equation_number, block.caption_id is not None)


def _draft(block: HypBlock, index: int, other: list[HypBlock]) -> tuple[dict[str, Any], list[str]]:
    fallback = "\n".join(item.text for item in other)
    text = block.text or (fallback if block.kind != "figure" else "")
    reading: dict[str, Any] = {
        "block_id": f"b{index + 1:04}",
        "kind": block.kind,
        "region": asdict(_box(block)),
        "reading_order": index,
        "text": text,
    }
    if block.kind == "formula":
        return reading, _draft_formula(reading, block, text)
    if block.kind == "table":
        return reading, _draft_table(reading, block, text)
    return reading, []


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


def merge_readings(
    primary: PageSubmission, alternate: PageSubmission
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use the layout reading for boundaries; show alternate readings and bring in its omissions."""
    first = tuple(sorted(primary.blocks, key=lambda block: block.reading_order))
    second = tuple(sorted(alternate.blocks, key=lambda block: block.reading_order))
    groups, missing = _group(first, second)
    blocks: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    previous_rank = -1
    for block in (*first, *missing):
        others = groups.get(block.block_id, []) if block in first else []
        position = min((item.reading_order for item in others), default=previous_rank)
        flags = _flags(block, others, previous_rank, position)
        reading, structure_flags = _draft(block, len(blocks), others)
        blocks.append(reading)
        notes.append(
            {
                "block_id": reading["block_id"],
                "flags": flags + structure_flags,
                "alternate": [
                    {"kind": item.kind, "text": item.text, "region": asdict(_box(item))}
                    for item in others
                ]
                if flags
                else [],
            }
        )
        previous_rank = max(previous_rank, position)
    _link_figures(blocks, notes)
    return blocks, [note for note in notes if note["flags"]]


def printed_page(blocks: list[dict[str, Any]]) -> str | None:
    """Only suggest a folio when one labelled numeric block exists."""
    options = [block["text"].strip() for block in blocks if block["kind"] == "page-number"]
    return str(options[0]) if len(options) == 1 and options[0].isdecimal() else None
