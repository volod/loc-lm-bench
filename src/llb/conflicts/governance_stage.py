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
semantic tier produced anyway, and off the rows the audit returned.

ONE pair is named, because the deliverable is the KNOB and the knob is a property of the stage
rather than of the pair. Which pair that is, is the EARLIEST stage present rather than the first
pair in corpus order: corpus order names whichever pair's documents happen to sort first, which on
a corpus with one genuine chunking gap and many merely unrelated pairs is a `candidates` pair that
was never going to pair -- and the gap goes unmentioned. The earliest stage is the one an operator
has to fix first anyway, since a document the store never held cannot have met a threshold.
"""

from collections.abc import Container, Sequence

from llb.conflicts.document_chunks import DocumentChunks
from llb.conflicts.document_exclusions import DocumentExclusions
from llb.conflicts.governance import ORDERING_FIELDS, compare_editions, edition_key
from llb.core.contracts.common import JsonObject

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

# The order a PAIR meets the stages, which is how a pair whose two documents stopped at different
# ones is attributed: the earlier stop explains the pair, because a document the store never held
# cannot have met a threshold. `effort` is absent -- it is a property of the RUN, not of a document.
PAIR_STAGE_ORDER = (
    STAGE_CHUNKING,
    STAGE_DUPLICATE_COLLAPSE,
    STAGE_CLAIM_FLOOR,
    STAGE_CANDIDATES,
)
# The order the stages are REPORTED in, which is the same order with the one knobless stage sunk to
# the end. Duplicate collapse is not a loss an operator acts on -- the claim is in the store the
# whole time under the copy that survived -- so it can never outrank a stage that names real work;
# it is reported only when it is the only thing a run lost. Derived from `STAGE_KNOBS` rather than
# written out, so a stage added with a knob takes its pipeline position automatically.
REPORT_STAGE_ORDER = tuple(stage for stage in PAIR_STAGE_ORDER if stage in STAGE_KNOBS) + tuple(
    stage for stage in PAIR_STAGE_ORDER if stage not in STAGE_KNOBS
)


def _quoted(doc_ids: Sequence[str], joiner: str) -> str:
    return f" {joiner} ".join(f"`{doc_id}`" for doc_id in doc_ids)


def _document_stage(doc_id: str, chunks: DocumentChunks) -> str:
    """The stage a DOCUMENT is lost at, whatever it is paired with.

    Every stage below `effort` is a property of ONE document, which is what keeps the scan off a
    quadratic path: the documents that can demonstrate a stage are found in a single pass over the
    corpus, and only their own pairs are ever tested.
    """
    if not chunks.stored.get(doc_id):
        return STAGE_DUPLICATE_COLLAPSE if chunks.stored_copy_of(doc_id) else STAGE_CHUNKING
    if not chunks.comparable.get(doc_id):
        return STAGE_CLAIM_FLOOR
    return STAGE_CANDIDATES


def _collapsed_stage(missing: Sequence[str], chunks: DocumentChunks) -> tuple[str, str, str]:
    """The pair is missing only documents the store already holds under a copy's name.

    Not a loss and not a knob: re-chunking or rebuilding would collapse the duplicate again, and
    the claim is in the store the whole time under the copy that survived.
    """
    kept = [chunks.stored_copy_of(doc_id) for doc_id in missing]
    surviving = _quoted(sorted({copy for copy in kept if copy}), "and")
    proved = "them copies" if len(missing) > 1 else "it a copy"
    return (
        STAGE_DUPLICATE_COLLAPSE,
        f"no chunk of {_quoted(missing, 'or')} is in the store the audit read, because the hash "
        f"tier proved {proved} of {surviving}, whose chunks the store kept",
        f"none -- read this pair through {surviving} instead",
    )


def _floor_stage(
    filtered: Sequence[str], exclusions: DocumentExclusions | None
) -> tuple[str, str, str]:
    """Every chunk of these documents is excluded -- for which reason, and at which floor.

    Without the per-document exclusion record the reading can only offer the disjunction the run
    threw away ("front matter, below the floor, or a metadata block"), two thirds of which the named
    knob cannot move. With it, the reason carries the counts and the knob carries a VALUE -- or says
    that no value works, when nothing the floor governs is what excluded the document.
    """
    excluded = _quoted(filtered, "and")
    if exclusions is None:
        return (
            STAGE_CLAIM_FLOOR,
            f"every chunk of {excluded} in the store is excluded from comparison -- front matter, "
            "below `--min-claim-tokens`, or a repeated metadata block",
            STAGE_KNOBS[STAGE_CLAIM_FLOOR],
        )
    per_document = "; ".join(f"`{doc_id}`: {exclusions.phrase_for(doc_id)}" for doc_id in filtered)
    floors = [exclusions.floor_for(doc_id) for doc_id in filtered]
    reason = f"every chunk of {excluded} in the store is excluded from comparison ({per_document})"
    if any(floor is None for floor in floors):
        return (
            STAGE_CLAIM_FLOOR,
            reason,
            "re-chunk so the claim lands in a body chunk of its own -- no `--min-claim-tokens` "
            "value returns this pair, because nothing the floor governs is what excluded it",
        )
    # The pair needs EVERY filtered side back, so the floor is the lowest of their recovery points.
    return (
        STAGE_CLAIM_FLOOR,
        reason,
        f"lower `--min-claim-tokens` to {min(floor for floor in floors if floor is not None)} or "
        f"below (this run used {exclusions.min_claim_tokens}), or re-chunk so the claim lands in a "
        "longer chunk",
    )


def _stage_of(
    pair: tuple[str, str],
    chunks: DocumentChunks | None,
    exclusions: DocumentExclusions | None = None,
) -> tuple[str, str, str]:
    """The stage this pair stopped at, the observation that places it there, and the one knob."""
    if chunks is None:
        return (
            STAGE_EFFORT,
            "this run read no store, so no chunk pair could become a candidate and only whole "
            "documents were ever compared",
            STAGE_KNOBS[STAGE_EFFORT],
        )
    stages = {doc_id: _document_stage(doc_id, chunks) for doc_id in pair}
    stage = min(stages.values(), key=PAIR_STAGE_ORDER.index)
    missing = [
        doc_id for doc_id in pair if stages[doc_id] in (STAGE_CHUNKING, STAGE_DUPLICATE_COLLAPSE)
    ]
    if stage == STAGE_DUPLICATE_COLLAPSE:
        return _collapsed_stage(missing, chunks)
    if stage == STAGE_CHUNKING:
        return (
            STAGE_CHUNKING,
            f"no chunk of {_quoted(missing, 'or')} is in the store the audit read",
            STAGE_KNOBS[STAGE_CHUNKING],
        )
    if stage == STAGE_CLAIM_FLOOR:
        return _floor_stage(
            [doc_id for doc_id in pair if stages[doc_id] == STAGE_CLAIM_FLOOR], exclusions
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


def _is_lost(
    keyed: Sequence[tuple[str, JsonObject]],
    pair: tuple[int, int],
    returned_pairs: Container[tuple[str, str]],
) -> bool:
    """Whether this index pair is orderable by edition and joined by no returned row."""
    (left, left_governance), (right, right_governance) = keyed[pair[0]], keyed[pair[1]]
    if tuple(sorted([left, right])) in returned_pairs:
        return False
    return compare_editions(left_governance, right_governance).newer_side is not None


def first_lost_orderable_pair(
    documents: Sequence[tuple[str, JsonObject]],
    returned_pairs: Container[tuple[str, str]],
) -> tuple[str, str] | None:
    """The first document pair `compare_editions` orders that no returned row joins, in corpus order.

    Pairs are scanned in corpus order and the scan STOPS at the first hit, so what it costs is the
    distance to that hit rather than the pair space. Documents carrying no ordering field at all
    are dropped first -- they cannot sit in an orderable pair, so an undated corpus empties the
    scan instead of walking it. This is the rule when every lost pair stopped at the SAME stage,
    which is exactly the run that read no store; `earliest_lost_orderable_pair` is the rule
    otherwise.
    """
    keyed = _orderable_documents(documents)
    for left in range(len(keyed)):
        for right in range(left + 1, len(keyed)):
            if _is_lost(keyed, (left, right), returned_pairs):
                return (keyed[left][0], keyed[right][0])
    return None


def _first_pair_at(
    stage: str,
    keyed: Sequence[tuple[str, JsonObject]],
    members: Sequence[int],
    returned_pairs: Container[tuple[str, str]],
    chunks: DocumentChunks,
) -> tuple[int, int] | None:
    """The corpus-first lost pair the pair rule attributes to `stage`, tested from `members` only.

    A pair can only reach `stage` through a document that is lost there, so the members are the
    whole search space and the cost is one pass over the corpus PER MEMBER -- linear in the corpus
    for every stage but `candidates`, whose members are the documents that reached the candidate
    list and whose sweep is the one the corpus-order scan already paid.

    Every hit is confirmed against `_stage_of`, which stays the single implementation of the rule:
    a member's pair can belong to an EARLIER stage than the member does (a chunkless partner beats
    a filtered one), and such a pair is found under that stage instead.
    """
    best: tuple[int, int] | None = None
    for index in members:
        for other in range(len(keyed)):
            pair = (min(index, other), max(index, other))
            if index == other or not _is_lost(keyed, pair, returned_pairs):
                continue
            if _stage_of((keyed[pair[0]][0], keyed[pair[1]][0]), chunks)[0] != stage:
                continue
            # Partners ascend, so the first hit is this member's own corpus-first pair.
            best = pair if best is None else min(best, pair)
            break
    return best


def earliest_lost_orderable_pair(
    documents: Sequence[tuple[str, JsonObject]],
    returned_pairs: Container[tuple[str, str]],
    chunks: DocumentChunks | None,
) -> tuple[str, str] | None:
    """The lost orderable pair from the EARLIEST stage the run lost one at, in corpus order within it.

    A run that read no store lost every pair at the effort dial, so there is no earliest stage to
    pick and corpus order is the whole rule.
    """
    if chunks is None:
        return first_lost_orderable_pair(documents, returned_pairs)
    keyed = _orderable_documents(documents)
    document_stages = [_document_stage(doc_id, chunks) for doc_id, _ in keyed]
    for stage in REPORT_STAGE_ORDER:
        members = [index for index, at in enumerate(document_stages) if at == stage]
        if not members:
            continue
        found = _first_pair_at(stage, keyed, members, returned_pairs, chunks)
        if found is not None:
            return (keyed[found[0]][0], keyed[found[1]][0])
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


def attribution_over_returned(
    coverage: JsonObject,
    documents: Sequence[tuple[str, JsonObject]],
    returned_pairs: Container[tuple[str, str]],
    chunks: DocumentChunks | None,
    exclusions: DocumentExclusions | None = None,
) -> JsonObject:
    """The attribution against an explicit set of returned pairs -- the whole rule, once.

    Taking the returned pairs rather than the rows is what lets a bundle replay ask the same
    question at a DIFFERENT candidate budget (`candidate_record.py`): the budget changes which pairs
    came back and changes nothing else, so the rule must not be re-implemented to vary it.
    """
    if not int(coverage.get("orderable_document_pairs") or 0):
        return {}
    lost = earliest_lost_orderable_pair(documents, returned_pairs, chunks)
    if lost is None:
        return {}
    stage, reason, knob = _stage_of(lost, chunks, exclusions)
    return {
        LOST_PAIR_FIELD: {
            "documents": list(lost),
            "stage": stage,
            "reason": reason,
            "knob": knob,
        }
    }


def lost_pair_attribution(
    coverage: JsonObject,
    documents: Sequence[tuple[str, JsonObject]],
    rows: Sequence[JsonObject],
    chunks: DocumentChunks | None,
    exclusions: DocumentExclusions | None = None,
) -> JsonObject:
    """`{LOST_PAIR_FIELD: ...}` for the earliest stage a pair was lost at, or `{}` when none was.

    Empty is the common and correct answer twice over: a corpus that can order nothing has no pair
    to lose, and a run that returned every orderable pair it could have lost none -- in both cases
    there is no knob to name and the reading says nothing extra.
    """
    return attribution_over_returned(
        coverage, documents, returned_doc_pairs(rows), chunks, exclusions
    )


def lost_pair_sentence(coverage: JsonObject) -> str:
    """The attribution as one sentence beside the counts, or `""` when nothing was lost."""
    lost = coverage.get(LOST_PAIR_FIELD)
    if not isinstance(lost, dict):
        return ""
    left, right = lost["documents"]
    return (
        f"Earliest stage an orderable document pair was lost at: {STAGE_NAMES[lost['stage']]}, "
        f"shown by `{left}` + `{right}` ({lost['reason']}). One knob: {lost['knob']}."
    )
