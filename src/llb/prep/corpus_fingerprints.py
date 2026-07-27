"""Corpus and per-document fingerprints used by immutable RAG stores."""

import hashlib
import json
from pathlib import Path
from typing import Any

from llb.prep.pdf.model import PDF_CITATION_SUFFIX

CORPUS_MANIFEST = "corpus_manifest.json"


def corpus_fingerprint(corpus_root: Path | str) -> str:
    """Fingerprint corpus content, governance rows, citations, and conflict overlay."""
    root = Path(corpus_root)
    manifest = load_manifest(root)
    if manifest:
        items = manifest.get("items")
        rows = [
            _manifest_item_row(item)
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict) and item.get("status") == "ok"
        ]
        rows.sort(key=lambda row: str(row.get("doc_id")))
        fingerprint = _json_fingerprint(
            [_with_citation_sidecar(root, str(row.get("doc_id")), row) for row in rows]
        )
    else:
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                doc_id = path.relative_to(root).as_posix()
                row = {"path": doc_id, "sha256": _sha256_file(path)}
                files.append(_with_citation_sidecar(root, doc_id, row))
        fingerprint = _json_fingerprint(files)

    from llb.conflicts.overlay import load_applied_overlay, overlay_fingerprint

    overlay_sha = overlay_fingerprint(load_applied_overlay(root))
    return (
        fingerprint
        if overlay_sha is None
        else _json_fingerprint({"corpus": fingerprint, "conflict_overlay": overlay_sha})
    )


def corpus_doc_fingerprints(corpus_root: Path | str) -> dict[str, str]:
    """Return the per-document fingerprints consumed by refresh-index."""
    root = Path(corpus_root)
    manifest = load_manifest(root)
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
    else:
        out = {
            path.relative_to(root).as_posix(): _plain_doc_fingerprint(root, path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        }

    from llb.conflicts.overlay import load_applied_overlay, overlay_fingerprint_for_doc

    overlay = load_applied_overlay(root)
    for doc_id, fingerprint in list(out.items()):
        overlay_sha = overlay_fingerprint_for_doc(overlay, doc_id)
        if overlay_sha is not None:
            out[doc_id] = _json_fingerprint(
                {"document": fingerprint, "conflict_overlay": overlay_sha}
            )
    return out


def manifest_items_fingerprint(items: list[dict[str, Any]]) -> str:
    rows = [_manifest_item_row(item) for item in items if item.get("status") == "ok"]
    return _json_fingerprint(sorted(rows, key=lambda row: str(row.get("doc_id"))))


def load_manifest(corpus_root: Path) -> dict[str, Any] | None:
    path = corpus_root / CORPUS_MANIFEST
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_item_row(item: dict[str, Any]) -> dict[str, Any]:
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


def _citation_sidecar_sha(root: Path, doc_id: str) -> str | None:
    sidecar = root / Path(doc_id).with_suffix(PDF_CITATION_SUFFIX)
    return _sha256_file(sidecar) if sidecar.is_file() else None


def _with_citation_sidecar(root: Path, doc_id: str, row: dict[str, Any]) -> dict[str, Any]:
    sidecar_sha = _citation_sidecar_sha(root, doc_id)
    return row if sidecar_sha is None else {**row, "citations_sha256": sidecar_sha}


def _plain_doc_fingerprint(root: Path, path: Path) -> str:
    sha = _sha256_file(path)
    sidecar_sha = _citation_sidecar_sha(root, path.relative_to(root).as_posix())
    if sidecar_sha is None:
        return sha
    return _json_fingerprint({"sha256": sha, "citations_sha256": sidecar_sha})


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
