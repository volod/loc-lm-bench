"""What each corpus document reached in the store the semantic tier read.

Three facts per document, because a document misses at three different stages and each takes a
different knob: a document with no stored chunk never reached the store, a document whose stored
chunks are all excluded reached it and was filtered back out before any pair was formed, and a
document whose only copy in the store is another document's reached it under that copy's name.

The record is folded ONCE, where the comparable set is known exactly (`run_semantic_tiers`), and
read by the stage attribution in `governance_stage.py`. It is deliberately per DOCUMENT rather than
per pair: that is what lets the attribution find the documents that lost a pair in one pass over
the corpus instead of over its pairs.

It is also WRITTEN DOWN (`payload` / `from_payload`, carried in `summary.json` by
`stage_replay.py`), because it is the one input to the attribution that cannot be recovered from a
finished bundle: the store it was folded from is rebuilt, and a rebuilt store answers differently
while looking like the same recompute. The counts are three small maps keyed by the document's
position in the record's own id table (`document_index.py`) -- no chunk text and no ordinals, so the
record says what each document reached and nothing about what it said.
"""

from collections import Counter, defaultdict
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from llb.conflicts.document_index import (
    DocumentInterner,
    DocumentNaming,
    interned_counts,
    named_counts,
)
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord


def _copies(value: object, naming: DocumentNaming) -> dict[str, tuple[str, ...]]:
    """A recorded `{doc_id: [copy, ...]}` map; a bundle that recorded none has no copies."""
    if not isinstance(value, Mapping):
        return {}
    return {
        naming.name(str(key)): tuple(naming.name(str(copy)) for copy in names)
        for key, names in value.items()
        if isinstance(names, Sequence) and not isinstance(names, str)
    }


@dataclass(frozen=True)
class DocumentChunks:
    """Per-document chunk accounting: what the store holds, what the tier compared, what collapsed."""

    stored: Mapping[str, int]
    comparable: Mapping[str, int]
    # Documents the hash tier proved copies of each other, which is what tells a store that never
    # saw a document from a store that COLLAPSED it into a copy it kept.
    copies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        chunks: Sequence[ChunkRecord],
        comparable: Container[int],
        settled: Iterable[tuple[str, str]] = (),
    ) -> "DocumentChunks":
        """Fold the store's chunks, the tier's comparable ordinals, and the settled copies per document."""
        stored: Counter[str] = Counter()
        kept: Counter[str] = Counter()
        for ordinal, chunk in enumerate(chunks):
            doc_id = str(chunk["doc_id"])
            stored[doc_id] += 1
            if ordinal in comparable:
                kept[doc_id] += 1
        copies: dict[str, list[str]] = defaultdict(list)
        for left, right in settled:
            copies[left].append(right)
            copies[right].append(left)
        return cls(
            stored=dict(stored),
            comparable=dict(kept),
            copies={doc_id: tuple(sorted(named)) for doc_id, named in copies.items()},
        )

    def payload(self, interner: DocumentInterner) -> JsonObject:
        """The three maps as `summary.json` carries them, so a bundle can be re-read without a store.

        Every document is a POSITION in the record's own id table rather than an id, because these
        three maps mention the same corpus three times over (`document_index.py`).
        """
        return {
            "stored": interned_counts(self.stored, interner),
            "comparable": interned_counts(self.comparable, interner),
            "copies": {
                interner.key(doc_id): [interner.position(copy) for copy in named]
                for doc_id, named in self.copies.items()
            },
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], naming: DocumentNaming
    ) -> "DocumentChunks":
        """Read back what `payload` wrote. Absent is empty here, never an error.

        The ABSENCE that carries meaning is one level up: a run below the semantic tier records no
        chunk payload at all, and that is the `effort` reading. A payload present but missing a map
        is a document accounting with nothing in it, which is what a store with no chunks is.
        """
        return cls(
            stored=named_counts(payload.get("stored"), naming),
            comparable=named_counts(payload.get("comparable"), naming),
            copies=_copies(payload.get("copies"), naming),
        )

    def stored_copy_of(self, doc_id: str) -> str | None:
        """A document proved a copy of `doc_id` whose chunks the store DID keep, if there is one."""
        return next(
            (copy for copy in self.copies.get(doc_id, ()) if self.stored.get(copy)),
            None,
        )
