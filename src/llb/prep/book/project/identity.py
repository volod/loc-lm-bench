"""Stable document ids, and the previous version a changed text revises.

An unchanged ledger hashes to the same id. A changed text is a new file.
``revision_of`` names the previous file, which stays on disk.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from llb.prep.book.project.constants import ID_WIDTH, LOGICAL_KEY_FIELD
from llb.prep.corpus.governance import SOURCE_METADATA_SUFFIX


@dataclass(frozen=True, slots=True)
class PriorDocument:
    """One projection already written in the output directory."""

    doc_id: str
    logical_key: str
    text_sha256: str
    revision_of: str | None


def logical_key(source_sha256: str, node_id: str, part_index: int) -> str:
    """The identity of one content part, independent of its current text."""
    return f"{source_sha256}:{node_id}:{part_index}"


def text_sha256(text: str) -> str:
    """SHA-256 of the projected document bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id(key: str, digest: str) -> str:
    """The corpus filename for one part at one text."""
    prefix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:ID_WIDTH]
    return f"book-{prefix}-{digest[:ID_WIDTH]}.md"


def part_indexes(node_ids: list[str]) -> list[int]:
    """The part number of each file inside its structure node, in file order."""
    counts: dict[str, int] = {}
    indexes: list[int] = []
    for node_id in node_ids:
        part = counts.get(node_id, 0)
        counts[node_id] = part + 1
        indexes.append(part)
    return indexes


def load_priors(out_dir: Path) -> list[PriorDocument]:
    """Projections already in ``out_dir``. Files without a logical key are left alone."""
    if not out_dir.is_dir():
        return []
    found: list[PriorDocument] = []
    for path in sorted(out_dir.glob(f"*{SOURCE_METADATA_SUFFIX}")):
        prior = _prior(out_dir, path)
        if prior is not None:
            found.append(prior)
    return found


def revision_of(priors: list[PriorDocument], key: str, digest: str) -> str | None:
    """The document this text revises, or the link the same text already carries."""
    group = [prior for prior in priors if prior.logical_key == key]
    current = document_id(key, digest)
    same = next((prior for prior in group if prior.doc_id == current), None)
    if same is not None:
        return same.revision_of
    head = _head(group)
    if head is None:
        return None
    return head.doc_id


def _head(group: list[PriorDocument]) -> PriorDocument | None:
    if not group:
        return None
    referenced = {prior.revision_of for prior in group if prior.revision_of}
    heads = [prior for prior in group if prior.doc_id not in referenced]
    pool = heads or group
    return sorted(pool, key=lambda prior: prior.doc_id)[-1]


def _prior(out_dir: Path, path: Path) -> PriorDocument | None:
    if not path.name.endswith(".md" + SOURCE_METADATA_SUFFIX):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    key = payload.get(LOGICAL_KEY_FIELD)
    if not isinstance(key, str) or not key:
        return None
    doc_id = path.name[: -len(SOURCE_METADATA_SUFFIX)]
    text_path = out_dir / doc_id
    if not text_path.is_file():
        return None
    revision = payload.get("revision_of")
    return PriorDocument(
        doc_id=doc_id,
        logical_key=key,
        text_sha256=hashlib.sha256(text_path.read_bytes()).hexdigest(),
        revision_of=revision if isinstance(revision, str) and revision else None,
    )
