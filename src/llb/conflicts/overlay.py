"""Additive conflict overlay consumed by corpus chunking and fingerprints."""

import hashlib
import json
from pathlib import Path
from typing import Any

from llb.conflicts.constants import (
    APPLIED_OVERLAY_DIR,
    APPLIED_OVERLAY_FILE,
)
from llb.conflicts.resolution_policy import (
    ACTION_DROP_DUPLICATE,
    ACTION_PREFER_NEWER,
    STATUS_ACCEPTED,
)
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord


def applied_overlay_path(corpus_root: Path | str) -> Path:
    return Path(corpus_root) / APPLIED_OVERLAY_DIR / APPLIED_OVERLAY_FILE


def load_applied_overlay(corpus_root: Path | str) -> JsonObject | None:
    path = applied_overlay_path(corpus_root)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported conflict overlay schema")
    return payload


def directives_by_doc(overlay: JsonObject | None) -> dict[str, JsonObject]:
    if overlay is None:
        return {}
    raw = overlay.get("documents")
    if not isinstance(raw, dict):
        raise ValueError("conflict overlay documents must be an object")
    return {str(doc_id): value for doc_id, value in raw.items() if isinstance(value, dict)}


def overlay_fingerprint_for_doc(overlay: JsonObject | None, doc_id: str) -> str | None:
    directive = directives_by_doc(overlay).get(doc_id)
    if directive is None:
        return None
    encoded = json.dumps(directive, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def overlay_fingerprint(overlay: JsonObject | None) -> str | None:
    if overlay is None:
        return None
    encoded = json.dumps(overlay, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def apply_to_chunks(chunks: list[ChunkRecord], directive: JsonObject | None) -> list[ChunkRecord]:
    """Suppress overlapping retrieval units and attach non-destructive annotations."""
    if directive is None:
        return chunks
    if directive.get("suppress_document") is True:
        return []
    spans = directive.get("suppress_spans")
    suppress_spans = spans if isinstance(spans, list) else []
    kept = [chunk for chunk in chunks if not _suppressed(chunk, suppress_spans)]
    annotations = directive.get("annotations")
    if isinstance(annotations, list) and annotations:
        for chunk in kept:
            chunk["metadata"] = {
                **(chunk.get("metadata") or {}),
                "conflict_resolutions": [dict(row) for row in annotations if isinstance(row, dict)],
            }
    return kept


def _suppressed(chunk: ChunkRecord, spans: list[Any]) -> bool:
    start, end = int(chunk["char_start"]), int(chunk["char_end"])
    return any(
        isinstance(span, dict)
        and isinstance(span.get("char_start"), int)
        and isinstance(span.get("char_end"), int)
        and int(span["char_start"]) < end
        and start < int(span["char_end"])
        for span in spans
    )


def overlay_from_plan(plan: JsonObject) -> JsonObject:
    documents: dict[str, JsonObject] = {}
    items = plan.get("items")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        _add_item(documents, item)
    return {
        "schema_version": 1,
        "policy": plan.get("policy"),
        "source_findings_sha256": plan.get("source_findings_sha256"),
        "documents": dict(sorted(documents.items())),
    }


def _add_item(documents: dict[str, JsonObject], item: JsonObject) -> None:
    action = item.get("action")
    accepted = item.get("status") == STATUS_ACCEPTED
    target_side = item.get("target_side")
    if accepted and action in (ACTION_DROP_DUPLICATE, ACTION_PREFER_NEWER):
        target = item.get(target_side) if target_side in ("a", "b") else None
        if isinstance(target, dict):
            _add_suppression(documents, item, target)
    for side in ("a", "b"):
        ref = item.get(side)
        if isinstance(ref, dict) and isinstance(ref.get("doc_id"), str):
            directive = documents.setdefault(str(ref["doc_id"]), _empty_directive())
            directive["annotations"].append(
                {
                    "finding_id": item.get("finding_id"),
                    "relation": item.get("relation"),
                    "action": action,
                    "status": item.get("status"),
                }
            )


def _empty_directive() -> JsonObject:
    return {"suppress_document": False, "suppress_spans": [], "annotations": []}


def _add_suppression(
    documents: dict[str, JsonObject], item: JsonObject, target: JsonObject
) -> None:
    doc_id = str(target["doc_id"])
    directive = documents.setdefault(doc_id, _empty_directive())
    if item.get("tier") in ("hash", "lexical"):
        directive["suppress_document"] = True
        directive["suppress_spans"] = []
        return
    if directive.get("suppress_document") is True:
        return
    directive["suppress_spans"].append(
        {
            "char_start": int(target.get("char_start", 0)),
            "char_end": int(target.get("char_end", 0)),
            "finding_id": item.get("finding_id"),
            "action": item.get("action"),
        }
    )
