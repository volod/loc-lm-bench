"""Which STAGE a run lost an orderable document pair at -- one knob instead of four.

The coverage reading (`governance_coverage.py`) names RETRIEVAL when a corpus carries document
pairs `compare_editions` can order and the run returned none of them. RETRIEVAL is four knobs at
once -- raise `--effort`, raise `--max-candidate-pairs`, re-chunk, re-embed -- and an operator
told to try all four learns nothing about which one is theirs. The run already holds enough to
pick:

- no store was read, because the run stopped below the semantic tier: the EFFORT dial. The cheap
  tiers compare whole documents, so no chunk pair could have become a candidate.
- a side contributes no chunk to the store because a copy of it does: DUPLICATE COLLAPSE, and the
  only stage whose knob is "none" -- the claim is reachable through the copy the store kept, and
  rebuilding the store would collapse it again.
- a side contributes no chunk to the store and no copy of it does either: CHUNKING, or the store's
  ingestion. Nothing downstream could have seen the document.
- a side's chunks are in the store and none of them is comparable: the CLAIM-TOKEN FLOOR (with
  front matter and repeated metadata blocks, the other two exclusions on that path).
- both sides are comparable and the pair still never reached a returned row: CANDIDATE SELECTION,
  which is the cosine threshold and the candidate budget.

A REPORT over what the run already computed. Nothing here re-runs detection to chase the pair,
and nothing here moves a threshold or a budget: the stage is read off the chunk accounting the
semantic tier produced anyway, and off the rows the audit returned. One pair is named -- the first
that did not survive -- because the deliverable is the KNOB, and the knob is a property of the
stage rather than of the pair.
"""

from collections import Counter, defaultdict
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from llb.conflicts.governance import ORDERING_FIELDS, compare_editions, edition_key
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord

# The key the attribution rides under, inside the `governance_coverage` payload it explains.
LOST_PAIR_FIELD = "lost_orderable_pair"

STAGE_EFFORT = "effort"
STAGE_DUPLICATE_COLLAPSE = "duplicate_collapse"
STAGE_CHUNKING = "chunking"
STAGE_CLAIM_FLOOR = "claim_token_floor"
STAGE_CANDIDATES = "candidates"

# How each stage prints, and the ONE knob that stage is turned by. Kept together so a stage cannot
# be added with a reading and no advice, which is the state this module exists to end. Duplicate
# collapse has no fixed knob: its advice names the copy the store kept, so it is built per pair.
STAGE_NAMES = {
    STAGE_EFFORT: "the EFFORT DIAL",
    STAGE_DUPLICATE_COLLAPSE: "DUPLICATE COLLAPSE",
    STAGE_CHUNKING: "CHUNKING",
    STAGE_CLAIM_FLOOR: "the CLAIM-TOKEN FLOOR",
    STAGE_CANDIDATES: "CANDIDATE SELECTION",
}
STAGE_KNOBS = {
    STAGE_EFFORT: "raise `--effort` to `semantic` or `claim`",
    STAGE_CHUNKING: "rebuild the store over this corpus, or re-chunk it",
    STAGE_CLAIM_FLOOR: (
        "lower `--min-claim-tokens`, or re-chunk so the claim lands in a longer chunk"
    ),
    STAGE_CANDIDATES: "raise `--max-candidate-pairs`, or lower the cosine threshold",
}


@dataclass(frozen=True)
class DocumentChunks:
    """What each document reached, as the store the semantic tier read records it.

    Three facts per document, because they miss at three stages and take three different knobs: a
    document with no stored chunk never reached the store, a document whose stored chunks are all
    excluded reached it and was filtered back out before any pair was formed, and a document whose
    only copy in the store is another document's reached it under that copy's name.
    """

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


def _quoted(doc_ids: Sequence[str], joiner: str) -> str:
    return f" {joiner} ".join(f"`{doc_id}`" for doc_id in doc_ids)


def _collapsed_stage(missing: Sequence[str], chunks: DocumentChunks) -> tuple[str, str, str] | None:
    """The pair is missing only documents the store already holds under a copy's name.

    Not a loss and not a knob: re-chunking or rebuilding would collapse the duplicate again, and
    the claim is in the store the whole time under the copy that survived.
    """
    kept = [chunks.stored_copy_of(doc_id) for doc_id in missing]
    if not all(kept):
        return None
    surviving = _quoted(sorted({copy for copy in kept if copy}), "and")
    proved = "them copies" if len(missing) > 1 else "it a copy"
    return (
        STAGE_DUPLICATE_COLLAPSE,
        f"no chunk of {_quoted(missing, 'or')} is in the store the audit read, because the hash "
        f"tier proved {proved} of {surviving}, whose chunks the store kept",
        f"none -- read this pair through {surviving} instead",
    )


def _stage_of(pair: tuple[str, str], chunks: DocumentChunks | None) -> tuple[str, str, str]:
    """The stage this pair stopped at, the observation that places it there, and the one knob."""
    if chunks is None:
        return (
            STAGE_EFFORT,
            "this run read no store, so no chunk pair could become a candidate and only whole "
            "documents were ever compared",
            STAGE_KNOBS[STAGE_EFFORT],
        )
    missing = [doc_id for doc_id in pair if not chunks.stored.get(doc_id)]
    if missing:
        collapsed = _collapsed_stage(missing, chunks)
        return collapsed or (
            STAGE_CHUNKING,
            f"no chunk of {_quoted(missing, 'or')} is in the store the audit read",
            STAGE_KNOBS[STAGE_CHUNKING],
        )
    filtered = [doc_id for doc_id in pair if not chunks.comparable.get(doc_id)]
    if filtered:
        return (
            STAGE_CLAIM_FLOOR,
            f"every chunk of {_quoted(filtered, 'and')} in the store is excluded from comparison "
            "-- front matter, below `--min-claim-tokens`, or a repeated metadata block",
            STAGE_KNOBS[STAGE_CLAIM_FLOOR],
        )
    return (
        STAGE_CANDIDATES,
        "both sides carry comparable chunks in the store, so the pair was dropped where the "
        "candidate list is built",
        STAGE_KNOBS[STAGE_CANDIDATES],
    )


def _orderable_documents(
    documents: Sequence[tuple[str, JsonObject]],
) -> list[tuple[str, JsonObject]]:
    """Documents carrying a field an edition comparison could use -- the rest cannot sit in a pair."""
    return [
        (doc_id, governance)
        for doc_id, governance in documents
        if any(edition_key(governance, field) is not None for field in ORDERING_FIELDS)
    ]


def first_lost_orderable_pair(
    documents: Sequence[tuple[str, JsonObject]],
    returned_doc_pairs: Container[tuple[str, str]],
) -> tuple[str, str] | None:
    """The first document pair `compare_editions` orders that no returned row joins.

    Pairs are scanned in corpus order and the scan STOPS at the first hit, so what it costs is the
    distance to that hit rather than the pair space: a run that lost a pair early pays a handful of
    comparisons. Documents carrying no ordering field at all are dropped first -- they cannot sit
    in an orderable pair, so an undated corpus empties the scan instead of walking it.
    """
    keyed = _orderable_documents(documents)
    for index, (left, left_governance) in enumerate(keyed):
        for right, right_governance in keyed[index + 1 :]:
            if tuple(sorted([left, right])) in returned_doc_pairs:
                continue
            if compare_editions(left_governance, right_governance).newer_side is not None:
                return (left, right)
    return None


def returned_doc_pairs(rows: Sequence[JsonObject]) -> set[tuple[str, str]]:
    """The document pairs the audit RETURNED, as `findings.jsonl` rows carry them.

    A row's two sides carry their document's governance unchanged, so a returned row between two
    orderable documents is an orderable returned pair -- which is what makes "appears here" the
    whole survival test, with no second orderability check.
    """
    return {
        tuple(sorted([str(row["a"]["doc_id"]), str(row["b"]["doc_id"])]))  # type: ignore[misc]
        for row in rows
    }


def lost_pair_attribution(
    coverage: JsonObject,
    documents: Sequence[tuple[str, JsonObject]],
    rows: Sequence[JsonObject],
    chunks: DocumentChunks | None,
) -> JsonObject:
    """`{LOST_PAIR_FIELD: ...}` for the first orderable pair lost, or `{}` when none was.

    Empty is the common and correct answer twice over: a corpus that can order nothing has no pair
    to lose, and a run that returned every orderable pair it could have lost none -- in both cases
    there is no knob to name and the reading says nothing extra.
    """
    if not int(coverage.get("orderable_document_pairs") or 0):
        return {}
    lost = first_lost_orderable_pair(documents, returned_doc_pairs(rows))
    if lost is None:
        return {}
    stage, reason, knob = _stage_of(lost, chunks)
    return {
        LOST_PAIR_FIELD: {
            "documents": list(lost),
            "stage": stage,
            "reason": reason,
            "knob": knob,
        }
    }


def lost_pair_sentence(coverage: JsonObject) -> str:
    """The attribution as one sentence beside the counts, or `""` when nothing was lost."""
    lost = coverage.get(LOST_PAIR_FIELD)
    if not isinstance(lost, dict):
        return ""
    left, right = lost["documents"]
    return (
        f"First orderable document pair that did not survive: `{left}` + `{right}`, lost at "
        f"{STAGE_NAMES[lost['stage']]} ({lost['reason']}). One knob: {lost['knob']}."
    )
