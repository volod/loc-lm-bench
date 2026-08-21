"""Find the lost orderable document pair the stage attribution reports on.

A pair is LOST when `compare_editions` can order its two documents and no returned row joins them.
Which lost pair gets named is decided here; what that pair's stage and knob are is decided by
`governance_stage_rule`, which this module calls rather than re-implements.
"""

from collections.abc import Container, Sequence

from llb.conflicts.bundle.document_chunks import DocumentChunks
from llb.conflicts.governance.editions import ORDERING_FIELDS, compare_editions, edition_key
from llb.conflicts.governance.stage_rule import (
    REPORT_STAGE_ORDER,
    document_stage,
    stage_of,
)
from llb.core.contracts.common import JsonObject


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
            if stage_of((keyed[pair[0]][0], keyed[pair[1]][0]), chunks)[0] != stage:
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
    document_stages = [document_stage(doc_id, chunks) for doc_id, _ in keyed]
    for stage in REPORT_STAGE_ORDER:
        members = [index for index, at in enumerate(document_stages) if at == stage]
        if not members:
            continue
        found = _first_pair_at(stage, keyed, members, returned_pairs, chunks)
        if found is not None:
            return (keyed[found[0]][0], keyed[found[1]][0])
    return None
