"""Reuse unchanged conversions: fingerprint each source PDF by sha256 and rehydrate a previous
`ok` manifest item when the source bytes and the conversion request still match it.

Rehydration reconstructs the nested dataclasses (attempts / diagnostics / quality) from the stored
manifest payload, marking the item `reused`; a mismatch or a missing rendered file falls through to
a fresh conversion.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from llb.prep.pdf.model import (
    PARSER_AUTO,
    PDF_CORPUS_MANIFEST,
    SHA256_READ_CHUNK_BYTES,
    PdfCorpusItem,
    PdfDiagnostics,
    PdfExtractionQuality,
    PdfParserAttempt,
)
from dataclasses import fields


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(SHA256_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _previous_manifest_items(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the previous manifest of `out_dir` as `source -> item payload` (empty when absent)."""
    path = out_dir / PDF_CORPUS_MANIFEST
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items") if isinstance(payload, dict) else None
    previous: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("source"), str):
            previous[item["source"]] = item
    return previous


def _dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


def _nested_from_payload(cls: type, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    return cls(**_dataclass_kwargs(cls, payload))


def _attempt_from_payload(payload: dict[str, Any]) -> PdfParserAttempt:
    data = dict(payload)
    data["quality"] = _nested_from_payload(PdfExtractionQuality, payload.get("quality"))
    return PdfParserAttempt(**_dataclass_kwargs(PdfParserAttempt, data))


def _item_from_payload(payload: dict[str, Any]) -> PdfCorpusItem:
    data = dict(payload)
    data["attempts"] = [
        _attempt_from_payload(attempt)
        for attempt in (payload.get("attempts") or [])
        if isinstance(attempt, dict)
    ]
    data["diagnostics"] = _nested_from_payload(PdfDiagnostics, payload.get("diagnostics"))
    data["quality"] = _nested_from_payload(PdfExtractionQuality, payload.get("quality"))
    data["reused"] = True
    return PdfCorpusItem(**_dataclass_kwargs(PdfCorpusItem, data))


def _reusable_item(
    payload: dict[str, Any] | None,
    source_sha256: str,
    out_dir: Path,
    min_chars: int,
    parser: str,
) -> PdfCorpusItem | None:
    """Rehydrate a previous ok item when the source and conversion request still match it."""
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return None
    if payload.get("source_sha256") != source_sha256:
        return None
    doc_id = payload.get("doc_id")
    citation_path = payload.get("citation_path")
    selected_parser = payload.get("parser")
    n_chars = payload.get("n_chars")
    if not doc_id or not citation_path or not selected_parser:
        return None
    if parser != PARSER_AUTO and selected_parser != parser:
        return None
    if not isinstance(n_chars, int) or n_chars < min_chars:
        return None
    if not (out_dir / doc_id).is_file() or not (out_dir / citation_path).is_file():
        return None
    try:
        return _item_from_payload(payload)
    except (TypeError, ValueError):
        return None
