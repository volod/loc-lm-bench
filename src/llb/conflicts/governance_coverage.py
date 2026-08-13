"""Whether a corpus could HAVE a policy-choice delta at all -- the precondition behind a zero.

A zero policy-choice delta has two opposite readings and the delta alone cannot tell them apart:
the corpus may carry dated revisions the two policies happen to agree on, or it may carry no
governance dates at all -- in which case `superseded_by` can never be derived (`compare_editions`
needs `effective_date` or `version` on BOTH sides), the zero is a property of the INGESTION rather
than of the knowledge, and it is fixable where the corpus is built rather than where it is
reviewed. An operator reading "the choice is free here" off a bare zero is being told something
the run did not measure.

So the audit records the PRECONDITION beside the delta, at both levels it can be missing:

- **documents** carrying a field `compare_editions` can order on, over the whole corpus. This is
  the ingestion-side count, and a zero here means no run over this corpus can ever produce a
  non-zero delta, whatever its knowledge says.
- **returned pairs** whose two sides that same function actually orders. This is the stricter
  count and the one the reading turns on: a corpus can date every document and still return no
  orderable pair (two sides sharing one date order no better than two undated ones), and it is the
  returned pairs, not the documents, that a policy is replayed over.

Detection-side and policy-free: nothing here imports the resolution vocabulary, so the coverage is
recorded on every audit and only its READING beside a delta needs a policy to have been named.
The counts are the audit's own -- `compare_editions` is the same orderability test the tiers use to
promote a dated contradiction to `superseded_by`, so the precondition cannot drift from the thing
it is a precondition for.
"""

from collections.abc import Sequence

from llb.conflicts.census import counted
from llb.conflicts.governance import BASIS_EFFECTIVE_DATE, BASIS_VERSION, compare_editions
from llb.core.contracts.common import JsonObject

# 1 counts dated documents and orderable returned pairs beside the policy delta.
COVERAGE_SCHEMA_VERSION = 1
# The fields `compare_editions` can order on, named once so the count and the test cannot part.
ORDERING_FIELDS = (BASIS_EFFECTIVE_DATE, BASIS_VERSION)
ORDERING_FIELDS_PHRASE = " or ".join(f"`{field}`" for field in ORDERING_FIELDS)
COVERAGE_LABEL = "governance coverage"


def _field_value(governance: JsonObject, field: str) -> str:
    value = governance.get(field)
    return value.strip() if isinstance(value, str) else ""


def _side_governance(row: JsonObject, side: str) -> JsonObject:
    """One side's recorded governance, as `findings.jsonl` carries it."""
    value = row.get(side)
    governance = value.get("governance") if isinstance(value, dict) else None
    return governance if isinstance(governance, dict) else {}


def document_coverage(document_governance: Sequence[JsonObject]) -> JsonObject:
    """How many corpus documents carry a field an edition comparison could use.

    Per field as well as in total, because the two are fixed differently at ingestion: a corpus
    with `version` everywhere and no `effective_date` is orderable, and a corpus with neither is
    the one an operator has to go back to the source system for.
    """
    return {
        "documents": len(document_governance),
        "dated_documents": sum(
            1
            for governance in document_governance
            if any(_field_value(governance, field) for field in ORDERING_FIELDS)
        ),
        "documents_by_field": {
            field: sum(1 for governance in document_governance if _field_value(governance, field))
            for field in ORDERING_FIELDS
        },
    }


def pair_orderability(rows: Sequence[JsonObject]) -> JsonObject:
    """How many of the pairs the audit RETURNED `compare_editions` can actually order.

    Stricter than the document count and closer to what a policy sees: a pair is orderable only
    when both sides carry a field AND the two values differ, which is exactly the condition that
    promotes a dated contradiction to `superseded_by` -- the one relation the two shipped policies
    part on.
    """
    orderable = sum(
        1
        for row in rows
        if compare_editions(_side_governance(row, "a"), _side_governance(row, "b")).newer_side
        is not None
    )
    return {
        "returned_pairs": len(rows),
        "orderable_pairs": orderable,
        # None, never 0.0, on a run that returned no pair at all -- the same distinction
        # `moved_share` draws: nothing to order and nothing orderable among what was returned are
        # different answers.
        "orderable_share": round(orderable / len(rows), 6) if rows else None,
    }


def governance_coverage(
    document_governance: Sequence[JsonObject], rows: Sequence[JsonObject]
) -> JsonObject:
    """The whole precondition record: the corpus's dated documents and the run's orderable pairs."""
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        **document_coverage(document_governance),
        **pair_orderability(rows),
    }


def has_orderable_pair(coverage: JsonObject) -> bool:
    """Whether this run returned anything a policy choice could even have parted on."""
    return bool(coverage) and int(coverage.get("orderable_pairs") or 0) > 0


def coverage_counts_phrase(coverage: JsonObject) -> str:
    """`0 of 5 documents with effective_date or version, 0 of 100 returned pairs orderable`.

    Both clauses end in a noun phrase rather than a verb, so the line reads the same at a count of
    one as at a count of a hundred -- the counts here are read on corpora of both sizes.
    """
    documents = counted(int(coverage["documents"]), "document")
    fields = coverage.get("documents_by_field") or {}
    by_field = ", ".join(f"{fields.get(field, 0)} `{field}`" for field in ORDERING_FIELDS)
    pairs = counted(int(coverage["returned_pairs"]), "returned pair")
    return (
        f"{coverage['dated_documents']} of {documents} with {ORDERING_FIELDS_PHRASE} "
        f"({by_field}), {coverage['orderable_pairs']} of {pairs} orderable by `compare_editions`"
    )


def coverage_reading(coverage: JsonObject, *, zero_delta: bool) -> str:
    """The precondition beside the delta, plus the reading a ZERO delta cannot give itself.

    A zero with no orderable pair is STRUCTURAL -- the run could not have produced anything else,
    so it says nothing about whether the corpus's revisions agree, and it is fixed at ingestion.
    A zero with orderable pairs present is the other reading: the policies had dated pairs to part
    on and did not, which is evidence about the knowledge. A non-zero delta needs no reading; the
    counts still ride with it, because they are the pairs the choice was drawn from.
    """
    counts = f"{COVERAGE_LABEL}: {coverage_counts_phrase(coverage)}"
    if not zero_delta:
        return f"{counts} -- the rows the choice moves are drawn from those orderable pairs."
    if has_orderable_pair(coverage):
        return (
            f"{counts} -- so the zero above is about this corpus's KNOWLEDGE: a dated supersession "
            "was reachable on this run (these pairs carry the fields that promote one), and none "
            "of them became one the policies part over."
        )
    return (
        f"{counts} -- so the zero above is STRUCTURAL, not a finding about this corpus's "
        "knowledge: `superseded_by` is derived from `compare_editions`, no returned pair carries "
        "what it needs, and no policy pair could have differed on this run. Fixable at INGESTION "
        f"(record {ORDERING_FIELDS_PHRASE} on the documents), not at review."
    )
