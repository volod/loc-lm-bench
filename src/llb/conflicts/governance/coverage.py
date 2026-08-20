"""Whether a corpus could HAVE a policy-choice delta at all -- the precondition behind a zero.

A zero policy-choice delta has two opposite readings and the delta alone cannot tell them apart:
the corpus may carry dated revisions the two policies happen to agree on, or it may carry no
governance dates at all -- in which case `superseded_by` can never be derived (`compare_editions`
needs `effective_date` or `version` on BOTH sides), the zero is a property of the INGESTION rather
than of the knowledge, and it is fixable where the corpus is built rather than where it is
reviewed. An operator reading "the choice is free here" off a bare zero is being told something
the run did not measure.

So the audit records the PRECONDITION beside the delta, at every level it can be missing:

- **documents** carrying a field `compare_editions` can order on, over the whole corpus. This is
  the ingestion-side count, and a zero here means no run over this corpus can ever produce a
  non-zero delta, whatever its knowledge says.
- **document pairs** that same function can order, over the corpus's own documents. This is what
  the corpus COULD have supplied, and it needs no candidate list and no store -- a corpus where
  every document carries one shared date has dated documents and still nothing to order.
- **returned pairs** whose two sides that same function actually orders. This is the stricter
  count and the one the reading turns on: a corpus can date every document and still return no
  orderable pair (two sides sharing one date order no better than two undated ones), and it is the
  returned pairs, not the documents, that a policy is replayed over.

The middle count is what names the STAGE that lost an orderable pair, which is the whole reason it
is here: a corpus dated end to end whose one revision pair never entered the candidate list at the
budget the run used, and a corpus with no date anywhere, both return zero orderable pairs -- and an
operator fixes them in opposite places (raise the budget, re-chunk, or re-embed versus re-ingest).
Which of those knobs it is, on one named pair, comes from `governance_stage.py` and rides in this
payload as `lost_orderable_pair` -- optional, because a run that lost nothing has no knob to name.

Detection-side and policy-free: nothing here imports the resolution vocabulary, so the coverage is
recorded on every audit and only its READING beside a delta needs a policy to have been named.
The counts are the audit's own -- `compare_editions` is the same orderability test the tiers use to
promote a dated contradiction to `superseded_by`, so the precondition cannot drift from the thing
it is a precondition for.
"""

from collections import Counter
from collections.abc import Hashable, Sequence

from llb.conflicts.grouping.census import counted
from llb.conflicts.governance.editions import ORDERING_FIELDS, compare_editions, edition_key
from llb.conflicts.governance.stage import lost_pair_sentence
from llb.core.contracts.common import JsonObject

# 1 counts dated documents and orderable returned pairs beside the policy delta.
# 2 adds the orderable DOCUMENT pairs between them, which is what names the stage a run lost one at.
# 3 adds the optional `lost_orderable_pair` attribution, which names that stage on one pair.
COVERAGE_SCHEMA_VERSION = 3
ORDERING_FIELDS_PHRASE = " or ".join(f"`{field}`" for field in ORDERING_FIELDS)
# The knobs the retrieval reading lists when nothing narrowed them to one. `governance_stage.py`
# narrows them whenever the run recorded what it needs to; this is what an audit that did not says.
RETRIEVAL_KNOBS = " (raise `--effort` or `--max-candidate-pairs`, re-chunk, or re-embed)"
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


def _pairs(count: int) -> int:
    """How many unordered pairs `count` items make."""
    return count * (count - 1) // 2


def _same_key_pairs(keys: Sequence[Hashable]) -> int:
    """Pairs sharing a key, counted from the key multiset -- linear in the items, not the pairs."""
    return sum(_pairs(count) for count in Counter(keys).values())


def _differing_key_pairs(keys: Sequence[Hashable]) -> int:
    """Pairs whose keys differ. Every key here EXISTS; an absent one is filtered out first."""
    return _pairs(len(keys)) - _same_key_pairs(keys)


def _both_differing_pairs(keyed: Sequence[tuple[Hashable, Hashable]]) -> int:
    """Pairs differing in BOTH keys -- the overlap a per-field count would otherwise claim twice.

    From the multisets again: every pair, minus the ones agreeing on either key, plus back the ones
    agreeing on both, which those two subtractions each removed.
    """
    return (
        _pairs(len(keyed))
        - _same_key_pairs([first for first, _ in keyed])
        - _same_key_pairs([second for _, second in keyed])
        + _same_key_pairs(keyed)
    )


def document_pair_orderability(document_governance: Sequence[JsonObject]) -> JsonObject:
    """How many of the corpus's own DOCUMENT pairs `compare_editions` could order.

    Between the two counts around it, and the one that says which STAGE lost an orderable pair: it
    is measured on the corpus alone, so a run whose returned pairs order none while this count is
    non-zero lost the pair in RETRIEVAL -- candidate budget, chunking, or embedding -- rather than
    at ingestion, and re-ingesting the corpus would change nothing.

    Counted from the distinct ordering keys rather than by enumerating pairs, because a pair is
    orderable on a field exactly when both sides carry that field's key and the two keys differ.
    That makes the whole count inclusion-exclusion over two key multisets (orderable by date, plus
    orderable by version, minus the pairs both counts claim) and keeps a corpus of thousands of
    documents off a quadratic path. The unpack below is deliberate: two bases is what makes two
    terms enough, so a third ordering field must fail here rather than silently under-count.
    """
    date_field, version_field = ORDERING_FIELDS
    keyed = [
        (edition_key(governance, date_field), edition_key(governance, version_field))
        for governance in document_governance
    ]
    return {
        "document_pairs": _pairs(len(document_governance)),
        "orderable_document_pairs": (
            _differing_key_pairs([date for date, _ in keyed if date is not None])
            + _differing_key_pairs([version for _, version in keyed if version is not None])
            # Only a document carrying BOTH keys can sit in a pair the two terms above both count.
            - _both_differing_pairs([keys for keys in keyed if None not in keys])
        ),
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
    """The whole precondition record: the corpus's dated documents and pairs, and the run's pairs."""
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        **document_coverage(document_governance),
        **document_pair_orderability(document_governance),
        **pair_orderability(rows),
    }


def has_orderable_pair(coverage: JsonObject) -> bool:
    """Whether this run returned anything a policy choice could even have parted on."""
    return bool(coverage) and int(coverage.get("orderable_pairs") or 0) > 0


def has_orderable_document_pair(coverage: JsonObject) -> bool:
    """Whether the CORPUS could have supplied one, whatever the run's candidate list did with it."""
    return bool(coverage) and int(coverage.get("orderable_document_pairs") or 0) > 0


def coverage_counts_phrase(coverage: JsonObject) -> str:
    """`0 of 5 documents ..., 0 of 10 document pairs and 0 of 100 returned pairs orderable`.

    The two pair counts print beside each other because they are only ever read against each other:
    the difference between them is the stage that lost an orderable pair. Every clause ends in a
    noun phrase rather than a verb, so the line reads the same at a count of one as at a count of a
    hundred -- the counts here are read on corpora of both sizes.
    """
    documents = counted(int(coverage["documents"]), "document")
    fields = coverage.get("documents_by_field") or {}
    by_field = ", ".join(f"{fields.get(field, 0)} `{field}`" for field in ORDERING_FIELDS)
    document_pairs = counted(int(coverage["document_pairs"]), "document pair")
    pairs = counted(int(coverage["returned_pairs"]), "returned pair")
    return (
        f"{coverage['dated_documents']} of {documents} with {ORDERING_FIELDS_PHRASE} "
        f"({by_field}), {coverage['orderable_document_pairs']} of {document_pairs} and "
        f"{coverage['orderable_pairs']} of {pairs} orderable by `compare_editions`"
    )


def coverage_reading(coverage: JsonObject, *, zero_delta: bool) -> str:
    """The precondition beside the delta, plus the reading a ZERO delta cannot give itself.

    Three readings, because a zero delta fails in three places and only the first two used to be
    told apart:

    - orderable returned pairs present: the zero is about this corpus's KNOWLEDGE. The policies had
      dated pairs to part on and did not.
    - none returned but the CORPUS has orderable document pairs: the zero is structural for this
      run and the stage that lost the pair is RETRIEVAL. Re-ingesting fixes nothing here.
    - none at either level: structural and fixable at INGESTION, the only case where "record a
      date" is the advice.

    A non-zero delta needs no reading; the counts still ride with it, because they are the pairs
    the choice was drawn from.

    Every reading can carry the STAGE attribution, because a run that returned some orderable pairs
    can still have lost others -- and where the attribution is present it REPLACES the retrieval
    reading's list of knobs, which is the whole point of having measured which one it was.
    """
    counts = f"{COVERAGE_LABEL}: {coverage_counts_phrase(coverage)}"
    attribution = lost_pair_sentence(coverage)
    if not zero_delta:
        reading = f"{counts} -- the rows the choice moves are drawn from those orderable pairs."
    elif has_orderable_pair(coverage):
        reading = (
            f"{counts} -- so the zero above is about this corpus's KNOWLEDGE: a dated supersession "
            "was reachable on this run (these pairs carry the fields that promote one), and none "
            "of them became one the policies part over."
        )
    elif has_orderable_document_pair(coverage):
        reading = (
            f"{counts} -- so the zero above is STRUCTURAL for this RUN, and the stage that lost "
            "the orderable pair is RETRIEVAL, not ingestion: this corpus DOES carry document "
            "pairs `compare_editions` can order, and none of them reached the rows the audit "
            "returned. Fixable where the candidate list is built"
            f"{'' if attribution else RETRIEVAL_KNOBS}, not by re-ingesting the corpus."
        )
    else:
        reading = (
            f"{counts} -- so the zero above is STRUCTURAL, not a finding about this corpus's "
            "knowledge: `superseded_by` is derived from `compare_editions`, no document pair in "
            "the corpus carries what it needs (no date at all, or one edition on every document), "
            "and no policy pair could have differed on any run over this corpus. Fixable at "
            f"INGESTION (record {ORDERING_FIELDS_PHRASE} on the documents), not at review."
        )
    return f"{reading} {attribution}" if attribution else reading
