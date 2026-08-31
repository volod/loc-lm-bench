"""The lost-orderable-pair attribution: the payload, and the sentence that reads it.

The two halves it composes live next door -- `governance_stage_search` picks WHICH lost pair is
named, `governance_stage_rule` says which STAGE that pair stopped at and which single knob turns
it. This module is what the audit and the bundle replay call.
"""

from collections.abc import Collection, Sequence

from llb.conflicts.bundle.document_chunks import DocumentChunks
from llb.conflicts.bundle.document_exclusions import DocumentExclusions
from llb.conflicts.governance.stage_rule import STAGE_NAMES, stage_of
from llb.conflicts.governance.stage_census import lost_pairs_by_stage
from llb.conflicts.governance.stage_search import earliest_lost_orderable_pair
from llb.core.contracts.common import JsonObject

# The key the attribution rides under, inside the `governance_coverage` payload it explains.
LOST_PAIR_FIELD = "lost_orderable_pair"
LOST_PAIR_COUNTS_FIELD = "lost_pairs_by_stage"


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
    returned_pairs: Collection[tuple[str, str]],
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
    stage, reason, knob = stage_of(lost, chunks, exclusions)
    counts = lost_pairs_by_stage(documents, returned_pairs, chunks)
    return {
        LOST_PAIR_FIELD: {
            "documents": list(lost),
            "stage": stage,
            "reason": reason,
            "knob": knob,
            LOST_PAIR_COUNTS_FIELD: counts,
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
    sentence = (
        f"Earliest stage an orderable document pair was lost at: {STAGE_NAMES[lost['stage']]}, "
        f"shown by `{left}` + `{right}` ({lost['reason']}). One knob: {lost['knob']}."
    )
    counts = lost.get(LOST_PAIR_COUNTS_FIELD)
    if not isinstance(counts, dict) or not counts:
        return sentence
    total = sum(int(count) for count in counts.values())
    named = int(counts.get(str(lost["stage"]), 0))
    return (
        f"{sentence} Stage reach: {STAGE_NAMES[lost['stage']]} accounts for {named} of {total} "
        f"lost orderable pairs; full split: {lost_pair_counts_phrase(lost)}."
    )


def lost_pair_counts_phrase(lost: JsonObject) -> str:
    """`CHUNKING 2; CANDIDATE SELECTION 1`, or `""` for an older attribution."""
    counts = lost.get(LOST_PAIR_COUNTS_FIELD)
    if not isinstance(counts, dict):
        return ""
    return "; ".join(
        f"{STAGE_NAMES[stage]} {count}" for stage, count in counts.items() if int(count) > 0
    )
