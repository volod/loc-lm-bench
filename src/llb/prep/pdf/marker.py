"""Marker markdown extraction (optional layout-model parser)."""

import html
import importlib
from pathlib import Path
from typing import Any

from llb.prep.pdf.model import (
    _HTML_TAG,
    _MARKER_PAGE_ID,
    MARKER_TOOL,
    PdfExtraction,
    PdfPageChunk,
    _is_image_only_pdf,
    clean_pdf_text,
    inspect_pdf,
)


def _extract_with_marker(pdf_path: Path) -> PdfExtraction:
    """Extract with Marker when its optional package and API are installed."""
    try:
        pdf_module = importlib.import_module("marker.converters.pdf")
        models_module = importlib.import_module("marker.models")
        PdfConverter = pdf_module.PdfConverter
        create_model_dict = models_module.create_model_dict
    except ImportError as exc:
        raise RuntimeError("missing marker dependency") from exc
    try:
        force_ocr = _is_image_only_pdf(inspect_pdf(pdf_path))
        try:
            converter = PdfConverter(
                artifact_dict=create_model_dict(),
                config={"force_ocr": force_ocr, "paginate_output": True},
            )
        except TypeError:
            converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        pages = _marker_pages(rendered)
        markdown = getattr(rendered, "markdown", None)
        if markdown is None:
            markdown = "\n\n".join(page.text for page in pages if page.text) or str(rendered)
    except Exception as exc:
        raise RuntimeError(f"marker failed for {pdf_path.name}: {exc}") from exc
    return PdfExtraction(text=clean_pdf_text(str(markdown)), parser=MARKER_TOOL, pages=pages)


def _marker_pages(rendered: Any) -> list[PdfPageChunk]:
    root = _object_to_mapping(rendered)
    children = root.get("children") if root else getattr(rendered, "children", None)
    if not isinstance(children, list):
        return []

    pages: list[PdfPageChunk] = []
    for idx, child in enumerate(children):
        item = _object_to_mapping(child)
        block_type = str(item.get("block_type") or "").casefold()
        if block_type != "page":
            continue
        page_number = _marker_page_number(item, idx + 1)
        text = clean_pdf_text(_marker_block_text(item))
        if text:
            pages.append(
                PdfPageChunk(page=page_number, text=text, blocks=_marker_block_boxes(item))
            )
    return pages


def _object_to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _marker_page_number(item: dict[str, Any], fallback: int) -> int:
    item_id = str(item.get("id") or "")
    match = _MARKER_PAGE_ID.search(item_id)
    if match is None:
        return fallback
    return int(match.group(1)) + 1


def _marker_block_text(item: dict[str, Any]) -> str:
    children = item.get("children")
    if isinstance(children, list) and children:
        parts = [_marker_block_text(_object_to_mapping(child)) for child in children]
        return "\n\n".join(part for part in parts if part.strip())
    raw_html = str(item.get("html") or item.get("text") or "")
    return _html_to_text(raw_html)


def _marker_block_boxes(item: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    children = item.get("children")
    if not isinstance(children, list):
        return blocks
    for child in children:
        child_map = _object_to_mapping(child)
        polygon = child_map.get("polygon")
        blocks.append(
            {
                "class": str(child_map.get("block_type") or "unknown"),
                "bbox": polygon if isinstance(polygon, list) else None,
                "page_char_start": None,
                "page_char_end": None,
            }
        )
    return blocks


def _html_to_text(value: str) -> str:
    return html.unescape(_HTML_TAG.sub(" ", value)).strip()
