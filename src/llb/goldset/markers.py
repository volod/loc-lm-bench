"""Unresolved recovery markers inside a projected book document.

The book projection writes an uncertainty marker as a machine token, not as
prose. A gold span that overlaps one is not evidence: those characters were
never adopted from the scan.
"""

import json
import re
from pathlib import Path

from llb.prep.pdf.model import PDF_CITATION_SUFFIX

REFUSED_MARKER_REASONS = frozenset({"unresolved", "empty", "omission", "page-failed"})

MARKER = re.compile(
    r"\{\{llb-marker:(unresolved|empty|low-confidence|orphaned|omission|page-failed|blank)"
    r":([^:}]+):(\d+)-(\d+)\}\}"
)


def marker_token(reason: str, block_id: str, start: int, end: int) -> str:
    """One machine marker. The offsets are into the block's primary reading."""
    return f"{{{{llb-marker:{reason}:{block_id}:{start}-{end}}}}}"


def marker_overlap(text: str, start: int, end: int) -> str | None:
    """The unresolved token overlapping ``[start, end)``, if any."""
    for match in MARKER.finditer(text):
        if match.group(1) not in REFUSED_MARKER_REASONS:
            continue
        if match.start() < end and match.end() > start:
            return match.group(0)
    return None


def marker_error(
    item_id: str, doc_id: str, text: str, start: int, end: int, corpus_root: Path | None
) -> str | None:
    """The refusal for a span that overlaps an unresolved marker, if it does."""
    detail = marker_overlap(text, start, end)
    if detail is None and corpus_root is not None:
        detail = sidecar_marker_overlap(corpus_root, doc_id, start, end)
    if detail is None:
        return None
    return f"{item_id}: source span overlaps an unresolved marker in {doc_id} ({detail})"


def sidecar_marker_overlap(corpus_root: Path, doc_id: str, start: int, end: int) -> str | None:
    """An unresolved marker the citation sidecar records over ``[start, end)``."""
    path = corpus_root / Path(doc_id).with_suffix(PDF_CITATION_SUFFIX)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    pages = payload.get("pages")
    for page in pages if isinstance(pages, list) else []:
        found = _page_marker(page, start, end)
        if found is not None:
            return found
    return None


def _page_marker(page: object, start: int, end: int) -> str | None:
    if not isinstance(page, dict):
        return None
    blocks = page.get("blocks")
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, dict):
            continue
        found = _block_marker(block.get("markers"), start, end)
        if found is not None:
            return found
    return None


def _block_marker(markers: object, start: int, end: int) -> str | None:
    for marker in _marker_rows(markers):
        label = _refused_overlap(marker, start, end)
        if label is not None:
            return label
    return None


def _marker_rows(markers: object) -> list[dict[str, object]]:
    if not isinstance(markers, list):
        return []
    return [row for row in markers if isinstance(row, dict)]


def _refused_overlap(marker: dict[str, object], start: int, end: int) -> str | None:
    if marker.get("reason") not in REFUSED_MARKER_REASONS:
        return None
    span = _char_span(marker)
    if span is None or span[0] >= end or span[1] <= start:
        return None
    block_id = marker.get("block_id")
    label = block_id if isinstance(block_id, str) and block_id else "marker"
    return f"{label} {marker.get('reason')}"


def _char_span(marker: dict[str, object]) -> tuple[int, int] | None:
    char_start = marker.get("char_start")
    char_end = marker.get("char_end")
    if isinstance(char_start, int) and isinstance(char_end, int):
        return char_start, char_end
    return None
