"""MinerU middle JSON: one block per printed region, in normalized page coordinates."""

import json
from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import FormulaReading, TableReading
from llb.prep.book.engines.protocol import Proposed
from llb.prep.book.grid import formula_from_text, table_from_text

_KIND = {
    "text": "body",
    "ref_text": "body",
    "list": "body",
    "index": "body",
    "code": "body",
    "code_body": "body",
    "algorithm_body": "body",
    "doc_title": "heading",
    "paragraph_title": "heading",
    "header": "running-header",
    "footer": "running-header",
    "page_number": "page-number",
    "page_footnote": "footnote",
    "aside_text": "unknown",
    "equation": "formula",
    "image": "figure",
    "chart": "figure",
    "table": "table",
}
_UNIT_MAX = 1.5
_INLINE = {"text", "equation_inline", "code_inline", "hyperlink"}
_EXPAND = {"list", "index"}
_CAPTION = {"image_caption", "chart_caption", "code_caption"}


def parse_middle_file(path: Path, width: int, height: int) -> list[Proposed]:
    """Proposals from one middle-JSON file. An empty or foreign file is no proposals."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return parse_middle(payload, width, height)


def parse_middle(payload: object, width: int, height: int) -> list[Proposed]:
    """The first page of a middle document. Boxes are pixels on the rendered page."""
    if not isinstance(payload, dict) or not _is_middle(payload):
        return []
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], dict):
        return []
    blocks = pages[0].get("blocks")
    if not isinstance(blocks, list):
        return []
    proposed: list[Proposed] = []
    for block in blocks:
        proposed.extend(_items(block, len(proposed), width, height))
    return proposed


def _is_middle(payload: dict[str, object]) -> bool:
    marker = str(payload.get("schema") or payload.get("schema_id") or "")
    return "middle" in marker or isinstance(payload.get("pages"), list)


def _items(block: object, order: int, width: int, height: int) -> list[Proposed]:
    if not isinstance(block, dict):
        return []
    raw = str(block.get("type") or "")
    if raw in _EXPAND:
        nested = _nested(block, order, width, height)
        if nested:
            return nested
    item = _block(block, order, width, height)
    captions = _captions(block, order + (1 if item else 0), width, height)
    if item is None:
        return captions
    return [item, *captions]


def _nested(block: dict[str, object], order: int, width: int, height: int) -> list[Proposed]:
    content = block.get("content")
    if not isinstance(content, list):
        return []
    found: list[Proposed] = []
    for child in content:
        found.extend(_items(child, order + len(found), width, height))
    return found


def _captions(block: dict[str, object], order: int, width: int, height: int) -> list[Proposed]:
    content = block.get("content")
    if str(block.get("type") or "") not in {"image", "chart", "code"} or not isinstance(
        content, list
    ):
        return []
    found: list[Proposed] = []
    for child in content:
        if not isinstance(child, dict) or str(child.get("type") or "") not in _CAPTION:
            continue
        item = _block(child, order + len(found), width, height)
        if item is not None:
            found.append(item)
    return found


def _block(block: object, order: int, width: int, height: int) -> Proposed | None:
    if not isinstance(block, dict):
        return None
    raw = str(block.get("type") or "unknown")
    box = _pixels(block.get("bbox"), width, height)
    if box is None:
        return None
    if raw == "table":
        caption, body = _table_parts(block)
        return _make(
            "table", order, box, body, _formula("table", body), table_from_text(body, caption)
        )
    if raw in {"image", "chart", "image_body", "chart_body"}:
        return _make("figure", order, box, "", None, None)
    if raw in _CAPTION or raw == "table_caption":
        text = _plain(block.get("content"))
        if not text:
            return None
        return _make("caption", order, box, text, None, None)
    kind = _KIND.get(raw, "unknown")
    text = "" if kind == "figure" else _plain(block.get("content"))
    return _make(kind, order, box, text, _formula(kind, text), None)


def _table_parts(block: dict[str, object]) -> tuple[str | None, str]:
    caption: str | None = None
    body = ""
    content = block.get("content")
    if not isinstance(content, list):
        return None, _plain(content)
    for child in content:
        if not isinstance(child, dict):
            continue
        kind = str(child.get("type") or "")
        text = _plain(child.get("content"))
        if kind == "table_caption" and text:
            caption = text
        elif kind == "table_body":
            body = text
    return caption, body


def _plain(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    if content and all(
        isinstance(item, dict) and str(item.get("type") or "") in _INLINE for item in content
    ):
        return _inline(content)
    parts = []
    for item in content:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
        elif isinstance(item, dict):
            text = _plain(item.get("content"))
            if text:
                parts.append(text)
    return "\n".join(parts)


def _inline(spans: list[dict[str, object]]) -> str:
    chunks = []
    for item in spans:
        text = str(item.get("content") or "")
        if not text:
            continue
        if item.get("type") == "equation_inline":
            chunks.append(f"${text.strip()}$")
        else:
            chunks.append(text)
    return "".join(chunks).strip()


def _formula(kind: str, text: str) -> FormulaReading | None:
    if kind != "formula" or not text:
        return None
    return formula_from_text(text, display=True)


def _pixels(box: object, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in box)
    except (TypeError, ValueError):
        return None
    if max(x0, y0, x1, y1) <= _UNIT_MAX:
        x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _make(
    kind: str,
    order: int,
    box: tuple[float, float, float, float],
    text: str,
    formula: FormulaReading | None,
    table: TableReading | None,
) -> Proposed:
    return Proposed(kind, order, box[0], box[1], box[2], box[3], text, None, formula, table)
