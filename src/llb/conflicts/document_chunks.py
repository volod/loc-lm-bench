"""What each corpus document reached in the store the semantic tier read.

Three facts per document, because a document misses at three different stages and each takes a
different knob: a document with no stored chunk never reached the store, a document whose stored
chunks are all excluded reached it and was filtered back out before any pair was formed, and a
document whose only copy in the store is another document's reached it under that copy's name.

The record is folded ONCE, where the comparable set is known exactly (`run_semantic_tiers`), and
read by the stage attribution in `governance_stage.py`. It is deliberately per DOCUMENT rather than
per pair: that is what lets the attribution find the documents that lost a pair in one pass over
the corpus instead of over its pairs.
"""

from collections import Counter, defaultdict
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from llb.core.contracts.rag import ChunkRecord


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

    def stored_copy_of(self, doc_id: str) -> str | None:
        """A document proved a copy of `doc_id` whose chunks the store DID keep, if there is one."""
        return next(
            (copy for copy in self.copies.get(doc_id, ()) if self.stored.get(copy)),
            None,
        )
