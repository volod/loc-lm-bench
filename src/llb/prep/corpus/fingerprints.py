"""Corpus and per-document fingerprints used by immutable RAG stores."""

import hashlib
import json
from pathlib import Path
from typing import Any

from llb.prep.pdf.model import PDF_CITATION_SUFFIX

CORPUS_MANIFEST = "corpus_manifest.json"


def _manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = manifest.get("items")
    rows = [
        _manifest_item_row(item)
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and item.get("status") == "ok"
    ]
    return sorted(rows, key=lambda row: str(row.get("doc_id")))


def _corpus_content_fingerprint(root: Path, manifest: dict[str, Any] | None) -> str:
    if manifest:
        rows = [
            _with_citation_sidecar(root, str(row.get("doc_id")), row)
            for row in _manifest_rows(manifest)
        ]
        return _json_fingerprint(rows)
    rows = [
        _with_citation_sidecar(
            root,
            path.relative_to(root).as_posix(),
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)},
        )
        for path in sorted(root.rglob("*"))
        if _is_corpus_document(path)
    ]
    return _json_fingerprint(rows)


def _with_overlay_fingerprint(root: Path, fingerprint: str) -> str:
    from llb.conflicts.resolution.overlay import load_applied_overlay, overlay_fingerprint

    overlay_sha = overlay_fingerprint(load_applied_overlay(root))
    if overlay_sha is None:
        return fingerprint
    return _json_fingerprint({"corpus": fingerprint, "conflict_overlay": overlay_sha})


def corpus_fingerprint(corpus_root: Path | str) -> str:
    """Fingerprint corpus content, governance rows, citations, and conflict overlay."""
    root = Path(corpus_root)
    fingerprint = _corpus_content_fingerprint(root, load_manifest(root))
    return _with_overlay_fingerprint(root, fingerprint)


def _is_corpus_document(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".md", ".txt"}


def _manifest_doc_fingerprints(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["doc_id"]): _json_fingerprint(_with_citation_sidecar(root, str(row["doc_id"]), row))
        for row in _manifest_rows(manifest)
        if isinstance(row.get("doc_id"), str)
    }


def _plain_doc_fingerprints(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _plain_doc_fingerprint(root, path)
        for path in sorted(root.rglob("*"))
        if _is_corpus_document(path)
    }


def _apply_doc_overlay_fingerprints(root: Path, fingerprints: dict[str, str]) -> None:
    from llb.conflicts.resolution.overlay import load_applied_overlay, overlay_fingerprint_for_doc

    overlay = load_applied_overlay(root)
    for doc_id, fingerprint in list(fingerprints.items()):
        overlay_sha = overlay_fingerprint_for_doc(overlay, doc_id)
        if overlay_sha is not None:
            fingerprints[doc_id] = _json_fingerprint(
                {"document": fingerprint, "conflict_overlay": overlay_sha}
            )


def corpus_doc_fingerprints(corpus_root: Path | str) -> dict[str, str]:
    """Return the per-document fingerprints consumed by refresh-index."""
    root = Path(corpus_root)
    manifest = load_manifest(root)
    out = _manifest_doc_fingerprints(root, manifest) if manifest else _plain_doc_fingerprints(root)
    _apply_doc_overlay_fingerprints(root, out)
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
