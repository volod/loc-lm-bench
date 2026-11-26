"""Parsers for engines that return text: Markdown, dots.ocr JSON, and DeepSeek grounding boxes."""

import json
import re
from collections.abc import Iterator

from llb.core.contracts.data_prep.book_blocks import FormulaReading, TableReading
from llb.prep.book.engines.protocol import Proposed
from llb.prep.book.grid import formula_from_text, looks_like_caption, table_from_text

_FENCE = re.compile(r"^---\s*$")
_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<name>[^)]+)\)")
_BOX = re.compile(r"(?P<x>\d+)_(?P<y>\d+)_(?P<w>\d+)_(?P<h>\d+)")
_HEADING = re.compile(r"^(?P<marks>#{1,6})\s+(?P<text>.+)$")
_DISPLAY = re.compile(r"^\$\$(?P<body>.*)\$\$$")
_GROUND = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9_-]*)\[\[(?P<box>[^\]]+)\]\](?P<rest>.*)$")
_THOUSANDTHS = 1000.0
_MARK = "\x1f"
_Chunk = tuple[str, str, tuple[float, float, float, float] | None]


def parse_markdown(text: str, width: int, height: int) -> list[Proposed]:
    """Structural blocks. Boxes come from a figure filename; everything else is a vertical band."""
    chunks = list(_chunks(_body(text)))
    return _place(chunks, width, height)


def parse_dots(text: str, width: int, height: int) -> list[Proposed]:
    """dots.ocr layout JSON. A response that is not JSON falls back to Markdown."""
    payload = _json(text)
    items = _items(payload) if payload is not None else None
    if not items:
        return parse_markdown(text, width, height)
    proposed = []
    for order, item in enumerate(items):
        box = _quad(item)
        if box is None:
            continue
        label = str(item.get("category") or item.get("type") or item.get("label") or "unknown")
        content = str(item.get("text") or item.get("content") or "")
        proposed.append(_proposed(label, order, box, content))
    return proposed


def parse_grounding(text: str, width: int, height: int) -> list[Proposed]:
    """DeepSeek grounding lines `label[[x1, y1, x2, y2]]`, with the following lines as the text.

    The box is in thousandths of the page the model saw. It is converted to pixels here so a later
    pixel clamp does not also treat it as a 0..1000 box.
    """
    blocks = [_scale_thousandths(item, width, height) for item in _ground_blocks(text)]
    if not blocks:
        return parse_markdown(text, width, height)
    return _place(blocks, width, height)


def _body(text: str) -> str:
    lines = text.splitlines()
    if lines and _FENCE.match(lines[0]):
        for index, line in enumerate(lines[1:], start=1):
            if _FENCE.match(line):
                return "\n".join(lines[index + 1 :])
    return text


class _Scan:
    """The paragraph and table gathered from one markdown page."""

    def __init__(self) -> None:
        self.paragraph: list[str] = []
        self.table: list[str] = []
        self.caption: str | None = None
        self.html = False


def _chunks(text: str) -> Iterator[_Chunk]:
    scan = _Scan()
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "$$":
            yield from _flush(scan.paragraph, "body")
            body, index = _display_body(lines, index + 1)
            yield ("formula", body, None)
            continue
        if _table_line(stripped, scan.html):
            yield from _append_table(scan, stripped)
            index += 1
            continue
        yield from _finish_table(scan)
        yield from _text_line(scan, stripped)
        index += 1
    yield from _finish_table(scan)
    yield from _flush(scan.paragraph, "body")


def _append_table(scan: _Scan, stripped: str) -> Iterator[_Chunk]:
    """Add one table line. A closing HTML tag emits the table."""
    scan.caption = _remember_caption(scan.paragraph, scan.caption)
    yield from _flush(scan.paragraph, "body")
    scan.table.append(stripped)
    scan.html = scan.html or stripped.lower().startswith("<table")
    if scan.html and "</table>" in stripped.lower():
        yield from _finish_table(scan)


def _finish_table(scan: _Scan) -> Iterator[_Chunk]:
    """Emit an open table and clear it. A page with no open table yields nothing."""
    if not scan.table:
        return
    yield ("table", _marked(scan.table, scan.caption), None)
    scan.table.clear()
    scan.caption = None
    scan.html = False


def _text_line(scan: _Scan, stripped: str) -> Iterator[_Chunk]:
    """A blank line ends the paragraph. A special line is its own block."""
    if not stripped:
        yield from _flush(scan.paragraph, "body")
        return
    special = _special(stripped)
    if special is None:
        scan.paragraph.append(stripped)
        return
    yield from _flush(scan.paragraph, "body")
    yield special


def _display_body(lines: list[str], start: int) -> tuple[str, int]:
    """The lines between a `$$` pair. The returned index is the line after the closer."""
    body: list[str] = []
    index = start
    while index < len(lines) and lines[index].strip() != "$$":
        body.append(lines[index].strip())
        index += 1
    if index < len(lines):
        index += 1
    return "\n".join(line for line in body if line), index


def _flush(
    bucket: list[str], kind: str
) -> Iterator[tuple[str, str, tuple[float, float, float, float] | None]]:
    if bucket:
        yield (kind, "\n".join(bucket).strip(), None)
        bucket.clear()


def _special(stripped: str) -> tuple[str, str, tuple[float, float, float, float] | None] | None:
    """A line that is its own block. None when the line continues the current paragraph."""
    image = _IMAGE.search(stripped)
    if image:
        return ("figure", "", _image_box(image.group("name")))
    heading = _HEADING.match(stripped)
    if heading:
        return ("heading", heading.group("text").strip(), None)
    if _DISPLAY.match(stripped) or stripped.startswith("$$"):
        return ("formula", stripped.strip("$").strip(), None)
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 2:
        return ("inline-formula", stripped[1:-1].strip(), None)
    if stripped.startswith(r"\[") and stripped.endswith(r"\]"):
        return ("formula", stripped[2:-2].strip(), None)
    if stripped.startswith(r"\(") and stripped.endswith(r"\)"):
        return ("inline-formula", stripped[2:-2].strip(), None)
    return None


def _image_box(name: str) -> tuple[float, float, float, float] | None:
    match = _BOX.search(name)
    if match is None:
        return None
    x, y = float(match["x"]), float(match["y"])
    return (x, y, x + float(match["w"]), y + float(match["h"]))


def _place(
    chunks: list[tuple[str, str, tuple[float, float, float, float] | None]],
    width: int,
    height: int,
) -> list[Proposed]:
    missing = sum(1 for _, _, box in chunks if box is None)
    step = height / missing if missing else 0.0
    band = 0
    proposed = []
    for order, (kind, text, box) in enumerate(chunks):
        if box is None:
            y0 = band * step
            box = (0.0, y0, float(width), max(y0 + 1.0, (band + 1) * step))
            band += 1
        text, caption = _unmark(text)
        proposed.append(_proposed(kind, order, box, text, caption))
    return proposed


def _proposed(
    kind: str,
    order: int,
    box: tuple[float, float, float, float],
    text: str,
    caption: str | None = None,
) -> Proposed:
    return Proposed(
        kind,
        order,
        box[0],
        box[1],
        box[2],
        box[3],
        text,
        None,
        _formula(kind, text),
        _table(kind, text, caption),
    )


def _table_line(stripped: str, html: bool) -> bool:
    lowered = stripped.lower()
    if html:
        return True
    return stripped.startswith("|") or lowered.startswith("<table")


def _remember_caption(paragraph: list[str], caption: str | None) -> str | None:
    if len(paragraph) == 1 and looks_like_caption(paragraph[0]):
        return paragraph[0]
    return caption


def _marked(lines: list[str], caption: str | None) -> str:
    body = "\n".join(lines).strip()
    if not caption:
        return body
    return f"{_MARK}{caption}{_MARK}{body}"


def _unmark(text: str) -> tuple[str, str | None]:
    if not text.startswith(_MARK):
        return text, None
    _mark, caption, body = text.split(_MARK, 2)
    return body, caption or None


def _formula(kind: str, text: str) -> FormulaReading | None:
    key = kind.strip().lower().replace("_", "-")
    if key not in {"formula", "inline-formula", "equation", "display-formula"}:
        return None
    return formula_from_text(text, display=key != "inline-formula")


def _table(kind: str, text: str, caption: str | None) -> TableReading | None:
    key = kind.strip().lower().replace("_", "-")
    if key != "table":
        return None
    return table_from_text(text, caption)


def _json(text: str) -> object | None:
    start = text.find("[")
    brace = text.find("{")
    if start < 0 and brace < 0:
        return None
    cut = text[start if start >= 0 and (brace < 0 or start < brace) else brace :]
    try:
        parsed: object = json.loads(cut)
    except json.JSONDecodeError:
        return None
    return parsed


def _items(payload: object) -> list[dict[str, object]] | None:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("elements", "layout", "blocks", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return None


def _quad(item: dict[str, object]) -> tuple[float, float, float, float] | None:
    return _four(item.get("bbox") or item.get("box") or item.get("coordinate"))


def _four(raw: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4 or isinstance(raw[0], (list, tuple)):
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1)


def _csv_box(text: str) -> tuple[float, float, float, float] | None:
    return _four([part.strip() for part in text.split(",")])


def _scale_thousandths(
    item: tuple[str, str, tuple[float, float, float, float] | None], width: int, height: int
) -> tuple[str, str, tuple[float, float, float, float] | None]:
    kind, text, box = item
    if box is None or max(box) > _THOUSANDTHS:
        return item
    x0, y0, x1, y1 = box
    scaled = (
        x0 / _THOUSANDTHS * width,
        y0 / _THOUSANDTHS * height,
        x1 / _THOUSANDTHS * width,
        y1 / _THOUSANDTHS * height,
    )
    return (kind, text, scaled)


def _ground_blocks(
    text: str,
) -> Iterator[tuple[str, str, tuple[float, float, float, float] | None]]:
    label, parts, box = "", [], None
    found = False
    for line in text.splitlines():
        match = _GROUND.match(line.strip())
        if match is None:
            if label:
                parts.append(line)
            continue
        if label:
            found = True
            yield (label, "\n".join(parts).strip(), box)
        label = match.group("label")
        box = _csv_box(match.group("box"))
        parts = [match.group("rest").strip()] if match.group("rest").strip() else []
    if label:
        found = True
        yield (label, "\n".join(parts).strip(), box)
    if not found:
        return
