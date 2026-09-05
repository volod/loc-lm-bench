"""Additive conflict overlay consumed by corpus chunking and fingerprints."""

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Literal

from llb.conflicts.constants import (
    APPLIED_OVERLAY_DIR,
    APPLIED_OVERLAY_FILE,
)
from llb.conflicts.resolution.policy import (
    ACTION_DROP_DUPLICATE,
    ACTION_PREFER_NEWER,
    STATUS_ACCEPTED,
)
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.conflicts import (
    CONFLICT_OVERLAY_SCHEMA_ID,
    ConflictOverlay,
)
from llb.core.contracts.rag import ChunkRecord


# The overlay keeps the compact integer schema it has always written; the registry names the same
# single form. The two are one version in two encodings, so the map lives here, at the seam.
OVERLAY_LOCAL_SCHEMA_VERSION: Final[int] = 1
CONFLICT_OVERLAY_CONTRACT_VERSION: Final[Literal["1.0.0"]] = "1.0.0"


def applied_overlay_path(corpus_root: Path | str) -> Path:
    return Path(corpus_root) / APPLIED_OVERLAY_DIR / APPLIED_OVERLAY_FILE


def load_applied_overlay(corpus_root: Path | str) -> JsonObject | None:
    """The applied overlay at its current contract, or None where the corpus carries none.

    The overlay's own `schema_version` is an integer inside the file; the registry names the same
    form semantically, so an unsupported or future overlay is refused here -- before a store build
    folds it into a corpus fingerprint -- rather than after.
    """
    path = applied_overlay_path(corpus_root)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: conflict overlay must be an object")
    overlay = DEFAULT_REGISTRY.read_as(
        CONFLICT_OVERLAY_SCHEMA_ID,
        payload,
        version=overlay_contract_version(payload.get("schema_version")),
        source=str(path),
    )
    if not isinstance(overlay, ConflictOverlay):
        raise ValueError(f"{path}: overlay did not resolve to the current contract")
    return _local_form(overlay)


def _local_form(overlay: ConflictOverlay) -> JsonObject:
    """The overlay in the encoding the file uses, which is what a fingerprint hashes."""
    fields: JsonObject = overlay.model_dump(mode="json")
    fields.pop("schema_id")
    return {**fields, "schema_version": OVERLAY_LOCAL_SCHEMA_VERSION}


def overlay_contract_version(local_version: object) -> str:
    """The registry version naming the overlay's own integer schema."""
    if local_version != OVERLAY_LOCAL_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported conflict overlay schema {local_version!r}; "
            f"this build writes and reads {OVERLAY_LOCAL_SCHEMA_VERSION}"
        )
    return CONFLICT_OVERLAY_CONTRACT_VERSION


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
    """The overlay a resolution plan implies, built and validated through its contract."""
    documents: dict[str, JsonObject] = {}
    items = plan.get("items")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        _add_item(documents, item)
    overlay = ConflictOverlay(
        schema_id=CONFLICT_OVERLAY_SCHEMA_ID,
        schema_version=CONFLICT_OVERLAY_CONTRACT_VERSION,
        policy=_optional_str(plan.get("policy")),
        source_findings_sha256=_optional_str(plan.get("source_findings_sha256")),
        documents={doc: _ordered(directive) for doc, directive in sorted(documents.items())},
    )
    return _local_form(overlay)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _ordered(directive: JsonObject) -> JsonObject:
    """Sort a document's entries by their own identity, never by the order the rows arrived in.

    The overlay is folded into each document's fingerprint, so a plan built from the same findings
    in a different order must produce the same bytes -- otherwise re-reading an audit whose rows
    were merely re-sorted republishes a store generation that changes nothing.
    """
    directive["annotations"] = sorted(
        directive["annotations"], key=lambda entry: str(entry.get("finding_id"))
    )
    directive["suppress_spans"] = sorted(
        directive["suppress_spans"],
        key=lambda span: (int(span["char_start"]), int(span["char_end"]), str(span["finding_id"])),
    )
    return directive


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
