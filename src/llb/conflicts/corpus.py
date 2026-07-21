"""Corpus documents with content hashes and governance, for the conflict tiers.

`corpus_doc_fingerprints` deliberately folds the governance contract into each document's hash, so
two byte-identical documents that differ only in `effective_date` fingerprint differently -- which
is right for refresh (their chunk metadata must be rewritten) and wrong for duplicate detection.
This module therefore hashes CONTENT: `raw_sha` over the exact document text, and `normalized_sha`
over the Ukrainian-normalized token stream of the body with any front matter removed, so a
re-ingested document carrying a new edition date still reads as a duplicate of its predecessor.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.conflicts.hashing import sha256_text
from llb.prep.corpus_governance import (
    DEFAULT_SOURCE_SYSTEM,
    manifest_governance_by_doc,
    source_governance,
    split_front_matter,
)
from llb.rag.chunking.corpus import iter_docs
from llb.rag.lexical import tokenize


@dataclass(frozen=True)
class CorpusDoc:
    """One corpus document with the content hashes and governance the tiers compare."""

    doc_id: str
    text: str
    body_offset: int
    raw_sha: str
    normalized_sha: str
    governance: JsonObject

    @property
    def body(self) -> str:
        return self.text[self.body_offset :]

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _governance_for(
    corpus_root: Path, doc_id: str, text: str, from_manifest: dict[str, JsonObject]
) -> JsonObject:
    """Manifest governance when the corpus was ingested; otherwise read the document's own."""
    recorded = from_manifest.get(doc_id)
    if recorded:
        return dict(recorded)
    resolved = source_governance(
        corpus_root,
        corpus_root / doc_id,
        text=text,
        default_language=None,
        default_source_system=DEFAULT_SOURCE_SYSTEM,
        default_acl_label=None,
        ingestion_time="",
    )
    return {key: value for key, value in resolved.items() if value}


def load_corpus_docs(corpus_root: Path | str) -> list[CorpusDoc]:
    """Load every corpus document in canonical build order with hashes and governance."""
    root = Path(corpus_root)
    if not root.is_dir():
        raise SystemExit(f"[conflicts] corpus root does not exist: {root}")
    manifest_governance = {
        doc_id: dict(fields) for doc_id, fields in manifest_governance_by_doc(root).items()
    }
    docs: list[CorpusDoc] = []
    for doc_id, text in iter_docs(root):
        body, body_offset = split_front_matter(text)
        docs.append(
            CorpusDoc(
                doc_id=doc_id,
                text=text,
                body_offset=body_offset,
                raw_sha=sha256_text(text),
                normalized_sha=sha256_text(" ".join(tokenize(body))),
                governance=_governance_for(root, doc_id, text, manifest_governance),
            )
        )
    return docs


def whole_doc_span(doc: CorpusDoc) -> tuple[int, int]:
    """The claim span used when a finding is about a document as a whole."""
    return 0, doc.n_chars
