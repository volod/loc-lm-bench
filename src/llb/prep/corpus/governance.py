"""Governance metadata helpers for staged RAG corpora.

The fields here are additive provenance only. They never alter document text or character
offsets; chunking copies them into `ChunkRecord.metadata` so retrieval filters can enforce an
application-level ACL tag before generation sees any candidate.

Two lanes supply values. The operator lane authors them here -- CLI defaults, a
`<source>.metadata.json` sidecar, or markdown front matter. The acquisition lane renders them
upstream into the sidecar alone: front matter stays the operator-authored lane, so a projected
field name never has to be told apart from prose a document happens to open with.
"""

import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.prep.corpus.fingerprints import load_manifest
from llb.prep.corpus.governance_fields import (
    GOVERNANCE_FIELDS,
    LOCAL_GOVERNANCE_FIELDS,
    OPERATOR_GOVERNANCE_FIELDS,
)

SOURCE_METADATA_SUFFIX = ".metadata.json"
DEFAULT_SOURCE_SYSTEM = "local"
UNKNOWN_LANGUAGE = "und"

_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.S)
_KEY_VALUE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*?)\s*$")
_UKRAINIAN_CHARS = set("іїєґІЇЄҐ")
_CYRILLIC = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_LATIN = re.compile(r"[A-Za-z]")


def is_acquisition_source_system(value: object) -> bool:
    """Whether a source-system value names an upstream acquisition run.

    The projection contract reserves ``local`` for operator directories. Every non-empty,
    non-local value therefore opts that document into the acquired, append-only lane.
    """
    return isinstance(value, str) and bool(value.strip()) and value != DEFAULT_SOURCE_SYSTEM


def utc_ingestion_time() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def detect_language(text: str, default: str | None = None) -> str:
    """Cheap deterministic fallback for passthrough text when the operator supplies no language."""
    if default:
        return default
    if any(char in _UKRAINIAN_CHARS for char in text):
        return "uk"
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    if cyrillic:
        return "uk"
    if latin:
        return "en"
    return UNKNOWN_LANGUAGE


def source_governance(
    path: Path,
    *,
    text: str | None,
    default_language: str | None,
    default_source_system: str,
    default_acl_label: str | None,
    ingestion_time: str,
) -> dict[str, str | None]:
    """Return governance fields from defaults plus optional source-provided metadata.

    A source can provide `<name>.metadata.json` beside the document, or markdown-style front
    matter at the top of a text file. Source-provided values override operator defaults only for
    that document. The source text itself is passed through unchanged.

    Acquisition fields are read from the sidecar only; front matter carries operator fields.
    """
    supplied = _sidecar_metadata(path)
    if text is not None:
        supplied = {**_front_matter_metadata(text), **supplied}
    language = _string_or_none(supplied.get("language")) or detect_language(
        text or "", default_language
    )
    return _governance_row(
        supplied,
        language=language,
        default_source_system=default_source_system,
        default_acl_label=default_acl_label,
        ingestion_time=ingestion_time,
    )


def converted_governance(
    payload: Mapping[str, Any],
    *,
    default_language: str | None,
    default_source_system: str,
    default_acl_label: str | None,
    ingestion_time: str,
) -> dict[str, str | None]:
    """Governance for a lane whose own converter already emitted the fields (the PDF manifest).

    Unlike `source_governance` there is no source text to sniff a language from, so the operator
    default -- and then `und` -- stands in.
    """
    language = _string_or_none(payload.get("language")) or default_language or UNKNOWN_LANGUAGE
    return _governance_row(
        payload,
        language=language,
        default_source_system=default_source_system,
        default_acl_label=default_acl_label,
        ingestion_time=ingestion_time,
    )


def _governance_row(
    supplied: Mapping[str, Any],
    *,
    language: str,
    default_source_system: str,
    default_acl_label: str | None,
    ingestion_time: str,
) -> dict[str, str | None]:
    """One row over every governance field: supplied value, then operator default, then absent.

    Every field is present in the row, `None` where nothing supplied one -- a corpus that carries
    no acquisition provenance records its absence rather than omitting the key, so a reader never
    has to tell a missing field apart from an unasked question.
    """
    row: dict[str, str | None] = {
        field: _string_or_none(supplied.get(field)) for field in GOVERNANCE_FIELDS
    }
    row["language"] = language
    row["ingestion_time"] = ingestion_time
    row["source_system"] = row["source_system"] or default_source_system
    row["acl_label"] = row["acl_label"] or default_acl_label
    return row


def preserve_ingestion_time(
    previous: dict[str, Any] | None, governance: dict[str, str | None]
) -> dict[str, str | None]:
    """Keep the previous ingestion time when all non-time governance fields are unchanged."""
    if not isinstance(previous, dict):
        return governance
    prior_time = previous.get("ingestion_time")
    if not isinstance(prior_time, str):
        return governance
    for field in GOVERNANCE_FIELDS:
        if field in LOCAL_GOVERNANCE_FIELDS:
            continue
        if previous.get(field) != governance.get(field):
            return governance
    return {**governance, "ingestion_time": prior_time}


def item_governance(item: dict[str, Any]) -> dict[str, str | None]:
    return {field: _string_or_none(item.get(field)) for field in GOVERNANCE_FIELDS}


def manifest_governance_by_doc(corpus_root: Path | str) -> dict[str, dict[str, str | None]]:
    """Load ok manifest item governance keyed by `doc_id`; empty when no manifest exists."""
    manifest = load_manifest(Path(corpus_root))
    if not manifest:
        return {}
    out: dict[str, dict[str, str | None]] = {}
    items = manifest.get("items")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        doc_id = item.get("doc_id")
        if isinstance(doc_id, str):
            out[doc_id] = item_governance(item)
    return out


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + SOURCE_METADATA_SUFFIX)
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def split_front_matter(text: str) -> tuple[str, int]:
    """Split off any governance front matter: returns (body, body start offset).

    Front matter is metadata, not content, so content-level comparisons (corpus-conflict hashing)
    exclude it. Offsets stay anchored to the original text, which is never modified.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return text, 0
    return text[match.end() :], match.end()


def _front_matter_metadata(text: str) -> dict[str, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}
    out: dict[str, str] = {}
    for line in match.group("body").splitlines():
        key_match = _KEY_VALUE.match(line)
        if key_match and key_match.group(1) in OPERATOR_GOVERNANCE_FIELDS:
            out[key_match.group(1)] = key_match.group(2).strip().strip("\"'")
    return out


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
