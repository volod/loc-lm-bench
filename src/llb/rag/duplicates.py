"""Duplicate chunk collapse: index each distinct passage once, cite it everywhere.

Converted-PDF corpora repeat page furniture and table boilerplate verbatim, so the same chunk
text is indexed many times over. Two costs follow. The index budget pays for every copy, and --
because identical text embeds to an identical vector -- the copies score an EXACT tie, which the
backend breaks by candidate order: an item whose top-k cut falls inside such a tie has a metric
no retrieval property decides (see `llb.rag.noise_floor`).

Collapsing is loss-free for the metric and for citations because a survivor keeps every place its
text appears: the dropped copies are recorded as additive `metadata.duplicate_occurrences`, and
`occurrence_spans` re-expands them, so a retrieved survivor still hits a gold span labeled on any
copy and still resolves to every document that carries the passage.

The default tier is EXACT (byte-identical text); the coarser normalized tiers in
`llb.rag.duplicate_tiers` group texts that merely LOOK the same, and are opt-in because they merge
passages that genuinely differ. Near-duplicate documents remain a corpus-hygiene question owned by
the conflict lane, not a chunk-level one.
"""

from collections.abc import Sequence
from typing import NamedTuple, cast

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.duplicate_tiers import (
    TIER_EXACT,
    TIER_MASKED,
    TIER_NORMALIZED,
    duplicate_key,
)

# How a tier says "the same passage", for the one-line build summary.
TIER_SAMENESS = {
    TIER_EXACT: "byte-identical to",
    TIER_NORMALIZED: "normalization-equivalent to",
    TIER_MASKED: "digit-masked-equivalent to",
}

# Additive metadata keys on a survivor chunk. `OCCURRENCES_KEY` lists only the DROPPED copies --
# the survivor's own place is the record's own doc_id/offsets -- while `COUNT_KEY` counts all of
# them, so `duplicate_count > 1` marks a collapsed chunk.
OCCURRENCES_KEY = "duplicate_occurrences"
COUNT_KEY = "duplicate_count"


class DuplicateOccurrence(TypedDict):
    """One dropped copy of a survivor's text: its whole chunk record minus the identical text.

    Keeping the full record (offsets, ids, page/governance metadata) rather than only the span is
    what makes collapse REVERSIBLE -- `expand_duplicate_chunks` reconstructs the pre-collapse set
    exactly, which the incremental refresh needs -- and lets a citation quote the copy's own page.

    `text` is present ONLY at a coarser tier than `exact`, where the copy's text is merely
    equivalent to the survivor's rather than identical; carrying it keeps expansion exact there
    too. At the exact tier the key is never written, so records stay byte-identical to before the
    tiers existed.
    """

    doc_id: str
    char_start: int
    char_end: int
    chunk_id: NotRequired[str]
    parent_id: NotRequired[str]
    text: NotRequired[str]


class DuplicateStats(TypedDict):
    """Duplicate rate of a chunk set at one tier, before and after collapse.

    The intra/cross split is the census that says WHERE a corpus's repetition comes from: page
    furniture shared by many converted documents is cross-document, while boilerplate a single
    manual repeats section after section is intra-document and is a CONVERSION-side property of
    that one document (see `llb.prep.pdf.repeats`).
    """

    tier: NotRequired[str]  # absent in a store meta written before the tiers shipped (== exact)
    n: int  # chunks before collapse
    unique: int  # distinct texts AT THIS TIER == chunks after collapse
    collapsed: int  # copies removed (n - unique)
    duplicate_chunks: int  # chunks the tier calls the same passage as at least one other chunk
    duplicate_share: float  # their share of the chunk COUNT
    groups: int  # distinct texts appearing more than once
    largest_group: int  # copies in the largest group (1 when every text is distinct)
    intra_document_groups: int  # repeated groups whose copies all sit in ONE document
    cross_document_groups: int  # repeated groups whose copies span two or more documents


class _GroupShape(NamedTuple):
    """One distinct chunk text: how many copies it has and how many documents carry them."""

    size: int
    documents: int


class Collapse(NamedTuple):
    """Collapse result: the survivors, their input positions, and the measured duplicate rate.

    `kept` lets a caller carry any per-chunk parallel array (embedding rows, lexical entries)
    through the collapse without recomputing it.
    """

    chunks: list[ChunkRecord]
    kept: list[int]
    stats: DuplicateStats


def duplicate_stats(chunks: list[ChunkRecord], tier: str = TIER_EXACT) -> DuplicateStats:
    """Measure the duplicate rate of `chunks` at `tier` without changing them."""
    return _stats(_group_shapes(chunks, tier), tier)


def collapse_duplicate_chunks(chunks: list[ChunkRecord], tier: str = TIER_EXACT) -> Collapse:
    """Keep the FIRST chunk of every distinct text; fold the rest into its occurrence metadata.

    Build order is the corpus's sorted document order, so the survivor -- and therefore the
    persisted store -- is deterministic. Input records are never mutated: a collapsed survivor is
    a shallow copy with a new `metadata`, and an uncollapsed chunk is passed through untouched, so
    a corpus with no duplicates keeps byte-identical chunk records.

    `tier` (see `llb.rag.duplicate_tiers`) decides when two texts count as the same passage. At a
    coarser tier than `exact` a dropped copy's text differs from the survivor's, so the copy is
    recorded WITH its own text and expansion stays exact.
    """
    first_by_key: dict[str, int] = {}
    survivors: list[ChunkRecord] = []
    kept: list[int] = []
    occurrences: dict[int, list[DuplicateOccurrence]] = {}
    for position, chunk in enumerate(chunks):
        key = duplicate_key(chunk["text"], tier)
        index = first_by_key.get(key)
        if index is None:
            first_by_key[key] = len(survivors)
            survivors.append(chunk)
            kept.append(position)
            continue
        occurrences.setdefault(index, []).append(_occurrence(chunk, survivors[index]["text"]))
    shapes = [
        _survivor_shape(survivor, occurrences.get(index, ()))
        for index, survivor in enumerate(survivors)
    ]
    for index, copies in occurrences.items():
        survivors[index] = _with_occurrences(survivors[index], copies)
    return Collapse(chunks=survivors, kept=kept, stats=_stats(shapes, tier))


def expand_duplicate_chunks(
    chunks: list[ChunkRecord],
) -> tuple[list[ChunkRecord], list[int | None]]:
    """Undo collapse: every copy back as its own record, plus the row each copy was indexed under.

    A dropped copy is stored as its full record minus the text it shares with its survivor (and
    WITH its own text where a coarser tier merged texts that differ), so expansion is exact.
    Records come back grouped by document in ascending offset order -- the from-scratch build
    order -- so a caller can merge them per document.

    The second list maps each expanded record to the stored embedding row it may reuse: the
    survivor's row, which every copy of the SAME text shares, and `None` for a copy whose text
    differs from its survivor's -- its stored row encodes the survivor's wording, so reusing it
    for that copy would make a refreshed store drift from a rebuild.
    """
    expanded: list[tuple[ChunkRecord, int | None]] = []
    for row, chunk in enumerate(chunks):
        copies = duplicate_occurrences(chunk)
        if not copies:
            expanded.append((chunk, row))
            continue
        expanded.append((_without_occurrences(chunk), row))
        expanded.extend(
            (_restored(copy, chunk["text"]), row if "text" not in copy else None) for copy in copies
        )
    expanded.sort(key=lambda pair: (pair[0]["doc_id"], pair[0]["char_start"], pair[0]["char_end"]))
    return [record for record, _ in expanded], [row for _, row in expanded]


def occurrence_spans(chunk: ChunkRecord) -> list[SourceSpanRecord]:
    """Every place `chunk`'s text appears in the corpus: its own span plus its collapsed copies.

    The fast path (no collapsed copies) allocates nothing beyond the single-element list, which
    matters because span matching runs per retrieved chunk per item per noise-floor replicate.
    """
    copies = duplicate_occurrences(chunk)
    if not copies:
        return [cast(SourceSpanRecord, chunk)]
    return [cast(SourceSpanRecord, chunk), *(cast(SourceSpanRecord, copy) for copy in copies)]


def duplicate_occurrences(chunk: ChunkRecord) -> list[DuplicateOccurrence]:
    """The dropped copies recorded on a survivor chunk (empty for an uncollapsed chunk)."""
    metadata = chunk.get("metadata")
    if not metadata:
        return []
    copies = metadata.get(OCCURRENCES_KEY)
    return cast(list[DuplicateOccurrence], copies) if isinstance(copies, list) else []


def format_duplicate_stats(stats: DuplicateStats, collapsed: bool = True) -> str:
    """One ASCII line for a build summary (AGENTS.md: ASCII-only)."""
    sameness = TIER_SAMENESS.get(stats.get("tier", TIER_EXACT), TIER_SAMENESS[TIER_EXACT])
    measured = (
        f"duplicates: {stats['duplicate_chunks']}/{stats['n']} chunks "
        f"({stats['duplicate_share']:.1%}) {sameness} another, "
        f"{stats['groups']} groups{_census_clause(stats)}, "
        f"largest {stats['largest_group']} copies"
    )
    if collapsed:
        return f"{measured} -> {stats['unique']} indexed ({stats['collapsed']} collapsed)"
    return (
        f"{measured} -> all {stats['n']} indexed (--keep-duplicate-chunks; "
        f"identical text ties exactly, broken by chunk_id)"
    )


def _occurrence(chunk: ChunkRecord, survivor_text: str) -> DuplicateOccurrence:
    """The dropped chunk's whole record, minus a text identical to the survivor's.

    At the exact tier the texts are identical by construction and the key is dropped; at a coarser
    tier the copy keeps its own text, which is what makes expansion exact there as well.
    """
    kept = {key: value for key, value in chunk.items() if key != "text"}
    if chunk["text"] != survivor_text:
        kept["text"] = chunk["text"]
    return cast(DuplicateOccurrence, kept)


def _restored(occurrence: DuplicateOccurrence, survivor_text: str) -> ChunkRecord:
    """The dropped copy as a full chunk record: its own text where it kept one, else the shared."""
    return cast(ChunkRecord, {**occurrence, "text": occurrence.get("text", survivor_text)})


def _with_occurrences(survivor: ChunkRecord, copies: list[DuplicateOccurrence]) -> ChunkRecord:
    collapsed = cast(ChunkRecord, dict(survivor))
    collapsed["metadata"] = {
        **(survivor.get("metadata") or {}),
        OCCURRENCES_KEY: copies,
        COUNT_KEY: len(copies) + 1,
    }
    return collapsed


def _without_occurrences(survivor: ChunkRecord) -> ChunkRecord:
    """The survivor as it was chunked, with the collapse metadata removed."""
    restored = cast(ChunkRecord, dict(survivor))
    restored["metadata"] = {
        key: value
        for key, value in (survivor.get("metadata") or {}).items()
        if key not in (OCCURRENCES_KEY, COUNT_KEY)
    }
    return restored


def _census_clause(stats: DuplicateStats) -> str:
    """The intra/cross split, omitted for a store meta written before the census shipped."""
    intra = stats.get("intra_document_groups")
    cross = stats.get("cross_document_groups")
    if intra is None or cross is None:
        return ""
    return f" ({intra} intra-document, {cross} cross-document)"


def _survivor_shape(survivor: ChunkRecord, copies: Sequence[DuplicateOccurrence]) -> _GroupShape:
    """The group a survivor stands for: its own copy plus every copy folded into it."""
    documents = {survivor["doc_id"], *(copy["doc_id"] for copy in copies)}
    return _GroupShape(size=len(copies) + 1, documents=len(documents))


def _group_shapes(chunks: list[ChunkRecord], tier: str = TIER_EXACT) -> list[_GroupShape]:
    counts: dict[str, int] = {}
    documents: dict[str, set[str]] = {}
    for chunk in chunks:
        key = duplicate_key(chunk["text"], tier)
        counts[key] = counts.get(key, 0) + 1
        documents.setdefault(key, set()).add(chunk["doc_id"])
    return [_GroupShape(size, len(documents[key])) for key, size in counts.items()]


def _stats(shapes: list[_GroupShape], tier: str = TIER_EXACT) -> DuplicateStats:
    n = sum(shape.size for shape in shapes)
    unique = len(shapes)
    repeated = [shape for shape in shapes if shape.size > 1]
    duplicate_chunks = sum(shape.size for shape in repeated)
    return {
        "tier": tier,
        "n": n,
        "unique": unique,
        "collapsed": n - unique,
        "duplicate_chunks": duplicate_chunks,
        "duplicate_share": duplicate_chunks / n if n else 0.0,
        "groups": len(repeated),
        "largest_group": max((shape.size for shape in shapes), default=1),
        "intra_document_groups": sum(1 for shape in repeated if shape.documents == 1),
        "cross_document_groups": sum(1 for shape in repeated if shape.documents > 1),
    }
