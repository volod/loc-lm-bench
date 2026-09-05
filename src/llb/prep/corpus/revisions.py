"""Append-only revision retention for acquired corpus documents."""

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from llb.prep.corpus.governance import is_acquisition_source_system
from llb.prep.corpus.ingest_text import CorpusItem


def _corpus_item(payload: dict[str, Any]) -> CorpusItem:
    """Rehydrate one row written by this project's corpus manifest."""
    known = {field.name for field in fields(CorpusItem)}
    return CorpusItem(**{key: value for key, value in payload.items() if key in known})


def _previous_by_doc(previous: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(payload["doc_id"]): payload
        for payload in previous.values()
        if payload.get("status") == "ok" and isinstance(payload.get("doc_id"), str)
    }


def _revision_parents(items: list[CorpusItem]) -> list[str]:
    return [
        item.revision_of
        for item in items
        if item.revision_of and is_acquisition_source_system(item.source_system)
    ]


def _retained_item(out_dir: Path, revision_of: str, payload: dict[str, Any]) -> CorpusItem:
    staged_path = out_dir / revision_of
    if not staged_path.is_file():
        raise ValueError(
            f"cannot retain superseded document '{revision_of}': staged content is missing"
        )
    return replace(_corpus_item(payload), reused=True)


def retain_revision_ancestors(
    out_dir: Path,
    current: list[CorpusItem],
    previous: dict[str, dict[str, Any]],
) -> list[CorpusItem]:
    """Carry staged ancestors named by acquired revisions into the next manifest.

    The closure is transitive so replacing revision two with revision three does not discard
    revision one. A reference that predates this corpus is allowed; a row this corpus claims to
    retain must still have its staged bytes.
    """
    items = list(current)
    known_doc_ids = {item.doc_id for item in items if item.doc_id is not None}
    previous_by_doc = _previous_by_doc(previous)
    pending = _revision_parents(items)
    while pending:
        revision_of = pending.pop()
        if revision_of in known_doc_ids:
            continue
        payload = previous_by_doc.get(revision_of)
        if payload is None:
            continue
        retained = _retained_item(out_dir, revision_of, payload)
        items.append(retained)
        known_doc_ids.add(revision_of)
        pending.extend(_revision_parents([retained]))
    return items
