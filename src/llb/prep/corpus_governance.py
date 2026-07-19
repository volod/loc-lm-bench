"""Governance metadata helpers for staged RAG corpora.

The fields here are additive provenance only. They never alter document text or character
offsets; chunking copies them into `ChunkRecord.metadata` so retrieval filters can enforce an
application-level ACL tag before generation sees any candidate.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.prep.pdf.model import PDF_CITATION_SUFFIX

CORPUS_MANIFEST = "corpus_manifest.json"
GOVERNANCE_FIELDS = (
    "language",
    "version",
    "effective_date",
    "ingestion_time",
    "source_system",
    "acl_label",
)
SOURCE_METADATA_SUFFIX = ".metadata.json"
DEFAULT_SOURCE_SYSTEM = "local"
UNKNOWN_LANGUAGE = "und"

_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.S)
_KEY_VALUE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*?)\s*$")
_UKRAINIAN_CHARS = set("іїєґІЇЄҐ")
_CYRILLIC = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_LATIN = re.compile(r"[A-Za-z]")


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
    root: Path,
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
    """
    supplied = _sidecar_metadata(path)
    if text is not None:
        supplied = {**_front_matter_metadata(text), **supplied}
    language = _string_or_none(supplied.get("language")) or detect_language(
        text or "", default_language
    )
    return {
        "language": language,
        "version": _string_or_none(supplied.get("version")),
        "effective_date": _string_or_none(supplied.get("effective_date")),
        "ingestion_time": ingestion_time,
        "source_system": _string_or_none(supplied.get("source_system")) or default_source_system,
        "acl_label": _string_or_none(supplied.get("acl_label")) or default_acl_label,
    }


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
        if field == "ingestion_time":
            continue
        if previous.get(field) != governance.get(field):
            return governance
    return {**governance, "ingestion_time": prior_time}


def item_governance(item: dict[str, Any]) -> dict[str, str | None]:
    return {field: _string_or_none(item.get(field)) for field in GOVERNANCE_FIELDS}


def manifest_governance_by_doc(corpus_root: Path | str) -> dict[str, dict[str, str | None]]:
    """Load ok manifest item governance keyed by `doc_id`; empty when no manifest exists."""
    manifest = _load_manifest(Path(corpus_root))
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


def _manifest_item_row(item: dict[str, Any]) -> dict[str, Any]:
    """The canonical fingerprint row for one ok manifest item (content + governance contract)."""
    return {
        "source": item.get("source"),
        "doc_id": item.get("doc_id"),
        "kind": item.get("kind"),
        "n_chars": item.get("n_chars"),
        "source_sha256": item.get("source_sha256"),
        "language": item.get("language"),
        "version": item.get("version"),
        "effective_date": item.get("effective_date"),
        "source_system": item.get("source_system"),
        "acl_label": item.get("acl_label"),
    }


def corpus_fingerprint(corpus_root: Path | str) -> str:
    """Fingerprint the current corpus contract used by a store.

    Prefer `corpus_manifest.json` when present so source deletion and governance changes are
    visible. For hand-built corpora without a manifest, hash committed `.md`/`.txt` files.
    A document's `*.citations.json` sidecar (PDF page provenance) is part of its contract, so
    a sidecar-only regeneration moves the fingerprint too.
    """
    root = Path(corpus_root)
    manifest = _load_manifest(root)
    if manifest:
        items = manifest.get("items")
        rows = [
            _manifest_item_row(item)
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict) and item.get("status") == "ok"
        ]
        rows.sort(key=lambda row: str(row.get("doc_id")))
        return _json_fingerprint(
            [_with_citation_sidecar(root, str(row.get("doc_id")), row) for row in rows]
        )
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            doc_id = path.relative_to(root).as_posix()
            row = {"path": doc_id, "sha256": _sha256_file(path)}
            files.append(_with_citation_sidecar(root, doc_id, row))
    return _json_fingerprint(files)


def corpus_doc_fingerprints(corpus_root: Path | str) -> dict[str, str]:
    """Per-document fingerprints keyed by `doc_id` (the manifest-diff contract).

    Uses the same two sources as `corpus_fingerprint`: with `corpus_manifest.json` present each
    ok item's canonical row (content hash + governance contract) is hashed per `doc_id`; for
    hand-built corpora each committed `.md`/`.txt` file hashes to its sha256, keyed by its
    corpus-relative path -- the same `doc_id` chunking assigns. A doc's `*.citations.json`
    sidecar hash is folded in when one exists, so regenerated page spans count as a modified
    document (its chunks need re-annotating) while sidecar-less docs keep their plain hash.
    `refresh-index` diffs a store's recorded map against the current one to find added /
    modified / deleted documents.
    """
    root = Path(corpus_root)
    manifest = _load_manifest(root)
    if manifest:
        out: dict[str, str] = {}
        items = manifest.get("items")
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("status") != "ok":
                continue
            doc_id = item.get("doc_id")
            if isinstance(doc_id, str):
                out[doc_id] = _json_fingerprint(
                    _with_citation_sidecar(root, doc_id, _manifest_item_row(item))
                )
        return out
    return {
        path.relative_to(root).as_posix(): _plain_doc_fingerprint(root, path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    }


def _citation_sidecar_sha(root: Path, doc_id: str) -> str | None:
    """sha256 of the doc's PDF citation sidecar (`pdf-<digest>.citations.json`), or None."""
    sidecar = root / Path(doc_id).with_suffix(PDF_CITATION_SUFFIX)
    return _sha256_file(sidecar) if sidecar.is_file() else None


def _with_citation_sidecar(root: Path, doc_id: str, row: dict[str, Any]) -> dict[str, Any]:
    """Fold the citation-sidecar hash into a fingerprint row when the doc has a sidecar."""
    sidecar_sha = _citation_sidecar_sha(root, doc_id)
    return row if sidecar_sha is None else {**row, "citations_sha256": sidecar_sha}


def _plain_doc_fingerprint(root: Path, path: Path) -> str:
    """Hand-built-corpus doc fingerprint: the file sha256, plus the sidecar hash if present."""
    sha = _sha256_file(path)
    sidecar_sha = _citation_sidecar_sha(root, path.relative_to(root).as_posix())
    if sidecar_sha is None:
        return sha
    return _json_fingerprint({"sha256": sha, "citations_sha256": sidecar_sha})


def manifest_items_fingerprint(items: list[dict[str, Any]]) -> str:
    rows = [_manifest_item_row(item) for item in items if item.get("status") == "ok"]
    return _json_fingerprint(sorted(rows, key=lambda row: str(row.get("doc_id"))))


def _load_manifest(corpus_root: Path) -> dict[str, Any] | None:
    path = corpus_root / CORPUS_MANIFEST
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + SOURCE_METADATA_SUFFIX)
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _front_matter_metadata(text: str) -> dict[str, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}
    out: dict[str, str] = {}
    for line in match.group("body").splitlines():
        key_match = _KEY_VALUE.match(line)
        if key_match and key_match.group(1) in GOVERNANCE_FIELDS:
            out[key_match.group(1)] = key_match.group(2).strip().strip("\"'")
    return out


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
