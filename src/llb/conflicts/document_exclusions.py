"""WHY each document's chunks are not comparable, and the floor value that would return them.

The stage attribution can already say a pair stopped at the CLAIM-TOKEN FLOOR, but the sentence it
prints there is a disjunction -- "front matter, below `--min-claim-tokens`, or a repeated metadata
block" -- because the run threw the three reasons away and kept one total. Two of those three are
not fixed by the knob the stage names: front matter is excluded before the floor is ever consulted,
and a repeated metadata block is excluded for being ABOVE it. So an operator told to lower
`--min-claim-tokens` on a document whose every chunk is front matter is being told to turn a knob
that cannot move, which is the same four-knob problem the stage attribution exists to end, one
level down.

Three counters per document end it, plus one more that turns the knob into a VALUE:

- `front_matter` / `low_content` / `metadata_block`: the excluded chunks per document, per reason.
  Disjoint by construction (`ContentSelection` assigns exactly one reason per excluded chunk), so
  they sum to what `stored - comparable` already says in `DocumentChunks`.
- `recovery_floor`: the largest claim-token count among the document's low-content chunks -- the
  highest `--min-claim-tokens` at which the document has a chunk in the candidate set again. Absent
  when the document has no low-content chunk at all, and that absence IS the reading: no floor value
  returns it.

What the record deliberately does NOT support is replaying a DIFFERENT floor over the whole corpus.
Repeated-metadata detection runs over the candidate set the floor produces, so a lower floor can
add chunks that then re-group other chunks out of comparison; the exact answer needs a token count
per CHUNK and a re-run of the grouping, and that is over the boundary this record keeps
(`bundle_readings.py` states it). `recovery_floor` is a per-document necessary condition, quoted as
one, not a corpus-wide replay.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from llb.conflicts.document_index import (
    DocumentInterner,
    DocumentNaming,
    interned_counts,
    named_counts,
)
from llb.conflicts.semantic_filter import ContentSelection
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord

FRONT_MATTER = "front_matter"
LOW_CONTENT = "low_content"
METADATA_BLOCK = "metadata_block"

# How each reason prints inside the stage sentence, in the order a chunk meets them.
REASON_NAMES: dict[str, str] = {
    FRONT_MATTER: "front matter",
    LOW_CONTENT: "below `--min-claim-tokens`",
    METADATA_BLOCK: "a repeated metadata block",
}

STORED_KEY = "min_claim_tokens"
FLOOR_KEY = "recovery_floor"


@dataclass(frozen=True)
class DocumentExclusions:
    """Per-document exclusion reasons plus the floor each document would come back at."""

    front_matter: Mapping[str, int] = field(default_factory=dict)
    low_content: Mapping[str, int] = field(default_factory=dict)
    metadata_block: Mapping[str, int] = field(default_factory=dict)
    recovery_floor: Mapping[str, int] = field(default_factory=dict)
    # The floor this run ran at, so the advice can say "lower it to N" rather than just "to N".
    min_claim_tokens: int = 0

    @classmethod
    def of(
        cls,
        chunks: Sequence[ChunkRecord],
        selection: ContentSelection,
        *,
        min_claim_tokens: int,
    ) -> "DocumentExclusions":
        """Fold one `select_content_chunks` pass into per-document counters and floors."""
        per_reason: dict[str, Counter[str]] = {
            FRONT_MATTER: Counter(),
            LOW_CONTENT: Counter(),
            METADATA_BLOCK: Counter(),
        }
        for reason, ordinals in (
            (FRONT_MATTER, selection.front_matter),
            (LOW_CONTENT, selection.low_content),
            (METADATA_BLOCK, selection.metadata_blocks),
        ):
            for ordinal in ordinals:
                per_reason[reason][str(chunks[ordinal]["doc_id"])] += 1
        floor: Counter[str] = Counter()
        for ordinal, tokens in selection.low_content_tokens.items():
            doc_id = str(chunks[ordinal]["doc_id"])
            floor[doc_id] = max(floor.get(doc_id, 0), tokens)
        return cls(
            front_matter=dict(per_reason[FRONT_MATTER]),
            low_content=dict(per_reason[LOW_CONTENT]),
            metadata_block=dict(per_reason[METADATA_BLOCK]),
            recovery_floor=dict(floor),
            min_claim_tokens=min_claim_tokens,
        )

    def payload(self, interner: DocumentInterner) -> JsonObject:
        """The four maps as `summary.json` carries them, keyed by reason then by corpus position."""
        return {
            FRONT_MATTER: interned_counts(self.front_matter, interner),
            LOW_CONTENT: interned_counts(self.low_content, interner),
            METADATA_BLOCK: interned_counts(self.metadata_block, interner),
            FLOOR_KEY: interned_counts(self.recovery_floor, interner),
            STORED_KEY: int(self.min_claim_tokens),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], naming: DocumentNaming
    ) -> "DocumentExclusions":
        """Read back what `payload` wrote; a missing map is a run that excluded nothing that way."""
        stored = payload.get(STORED_KEY)
        return cls(
            front_matter=named_counts(payload.get(FRONT_MATTER), naming),
            low_content=named_counts(payload.get(LOW_CONTENT), naming),
            metadata_block=named_counts(payload.get(METADATA_BLOCK), naming),
            recovery_floor=named_counts(payload.get(FLOOR_KEY), naming),
            min_claim_tokens=stored if isinstance(stored, int) else 0,
        )

    def reasons_for(self, doc_id: str) -> list[tuple[str, int]]:
        """The document's non-zero exclusion reasons, in the order a chunk meets them."""
        counts = {
            FRONT_MATTER: self.front_matter,
            LOW_CONTENT: self.low_content,
            METADATA_BLOCK: self.metadata_block,
        }
        return [
            (reason, counts[reason].get(doc_id, 0))
            for reason in REASON_NAMES
            if counts[reason].get(doc_id, 0)
        ]

    def floor_for(self, doc_id: str) -> int | None:
        """The highest `--min-claim-tokens` at which this document has a candidate chunk again."""
        return self.recovery_floor.get(doc_id)

    def phrase_for(self, doc_id: str) -> str:
        """`2 below `--min-claim-tokens`, 1 front matter` -- the counts, never the disjunction."""
        return ", ".join(
            f"{count} {REASON_NAMES[reason]}" for reason, count in self.reasons_for(doc_id)
        )
