"""Docling's JSON export: texts, tables, and pictures, with boxes in PDF points.

Docling's origin is the bottom left. The ledger's origin is the top left of the rendered page,
so a box is flipped and scaled from the page's point size onto the image.
"""

import json
from pathlib import Path

from llb.prep.book.engines.protocol import Proposed


def parse_docling_dir(directory: Path, width: int, height: int) -> list[Proposed]:
    """The first Docling document JSON under `directory`."""
    for path in sorted(directory.glob("**/*.json")):
        payload = _load(path)
        if isinstance(payload, dict) and (payload.get("texts") or payload.get("body")):
            return parse_docling(payload, width, height)
    return []


def parse_docling(payload: dict[str, object], width: int, height: int) -> list[Proposed]:
    """Blocks in body order when Docling recorded one, else texts, then tables, then pictures."""
    page_w, page_h = _page_size(payload)
    by_ref = _index(payload)
    ordered = _in_body_order(payload, by_ref)
    proposed = []
    for order, item in enumerate(ordered):
        box = _box(item, page_w, page_h, width, height)
        if box is None:
            continue
        kind, text = _content(item)
        proposed.append(Proposed(kind, order, box[0], box[1], box[2], box[3], text, None))
    return proposed


def _load(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _page_size(payload: dict[str, object]) -> tuple[float, float]:
    pages = payload.get("pages")
    if isinstance(pages, dict):
        for page in pages.values():
            size = page.get("size") if isinstance(page, dict) else None
            if isinstance(size, dict) and "width" in size and "height" in size:
                return float(size["width"]), float(size["height"])
    return (1.0, 1.0)


def _index(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for key, kind in (("texts", "text"), ("tables", "table"), ("pictures", "picture")):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for index, item in enumerate(rows):
            if isinstance(item, dict):
                ref = str(item.get("self_ref") or f"#/{key}/{index}")
                found[ref] = {**item, "_kind": kind}
    return found


def _in_body_order(
    payload: dict[str, object], by_ref: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """Body order, expanding group references. Falls back to document order when the body is empty."""
    groups = _groups(payload)
    ordered = [by_ref[ref] for ref in _walk(payload.get("body"), groups, set()) if ref in by_ref]
    return ordered or list(by_ref.values())


def _groups(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = payload.get("groups")
    found: dict[str, dict[str, object]] = {}
    if not isinstance(rows, list):
        return found
    for index, item in enumerate(rows):
        if isinstance(item, dict):
            found[str(item.get("self_ref") or f"#/groups/{index}")] = item
    return found


def _walk(node: object, groups: dict[str, dict[str, object]], seen: set[str]) -> list[str]:
    if not isinstance(node, dict):
        return []
    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return []
        seen.add(ref)
        if ref in groups:
            return _walk(groups[ref], groups, seen)
        return [ref]
    children = node.get("children")
    if not isinstance(children, list):
        return []
    found = []
    for child in children:
        found.extend(_walk(child, groups, seen))
    return found


def _content(item: dict[str, object]) -> tuple[str, str]:
    kind = str(item.get("label") or item.get("_kind") or "unknown")
    if item.get("_kind") == "picture":
        return "figure", ""
    if item.get("_kind") == "table":
        return "table", _table_text(item)
    return kind, str(item.get("text") or "")


def _table_text(item: dict[str, object]) -> str:
    data = item.get("data")
    grid = data.get("grid") if isinstance(data, dict) else None
    if not isinstance(grid, list):
        return str(item.get("text") or "")
    rows = []
    for row in grid:
        if isinstance(row, list):
            rows.append(" | ".join(_cell(cell) for cell in row))
    return "\n".join(rows)


def _cell(cell: object) -> str:
    if isinstance(cell, dict):
        return str(cell.get("text") or "")
    return str(cell or "")


def _box(
    item: dict[str, object], page_w: float, page_h: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    prov = item.get("prov")
    if not isinstance(prov, list) or not prov or not isinstance(prov[0], dict):
        return None
    bbox = prov[0].get("bbox")
    if not isinstance(bbox, dict):
        return None
    left, top, right, bottom = (float(bbox.get(key, 0)) for key in ("l", "t", "r", "b"))
    scale_x = width / page_w if page_w else 1.0
    scale_y = height / page_h if page_h else 1.0
    origin = str(bbox.get("coord_origin") or "BOTTOMLEFT")
    x0, x1 = left * scale_x, right * scale_x
    if origin == "BOTTOMLEFT":
        y0, y1 = (page_h - top) * scale_y, (page_h - bottom) * scale_y
    else:
        y0, y1 = top * scale_y, bottom * scale_y
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
