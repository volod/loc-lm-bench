"""Exact-duplicate chunk collapse: index each distinct passage once, cite it everywhere.

Converted-PDF corpora repeat page furniture and table boilerplate verbatim, so the same chunk
text is indexed many times over. Two costs follow. The index budget pays for every copy, and --
because identical text embeds to an identical vector -- the copies score an EXACT tie, which the
backend breaks by candidate order: an item whose top-k cut falls inside such a tie has a metric
no retrieval property decides (see `llb.rag.noise_floor`).

Collapsing is loss-free for the metric and for citations because a survivor keeps every place its
text appears: the dropped copies are recorded as additive `metadata.duplicate_occurrences`, and
`occurrence_spans` re-expands them, so a retrieved survivor still hits a gold span labeled on any
copy and still resolves to every document that carries the passage.

Collapse is exact-only (byte-identical text). Near-duplicate documents are a corpus-hygiene
question owned by the conflict lane, not a chunk-level one.
"""

from typing import NamedTuple, cast

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord

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
    """

    doc_id: str
    char_start: int
    char_end: int
    chunk_id: NotRequired[str]
    parent_id: NotRequired[str]


class DuplicateStats(TypedDict):
    """Exact-duplicate rate of a chunk set, before and after collapse."""

    n: int  # chunks before collapse
    unique: int  # distinct chunk texts == chunks after collapse
    collapsed: int  # copies removed (n - unique)
    duplicate_chunks: int  # chunks byte-identical to at least one other chunk
    duplicate_share: float  # their share of the chunk COUNT
    groups: int  # distinct texts appearing more than once
    largest_group: int  # copies in the largest identical group (1 when all texts are distinct)


class Collapse(NamedTuple):
    """Collapse result: the survivors, their input positions, and the measured duplicate rate.

    `kept` lets a caller carry any per-chunk parallel array (embedding rows, lexical entries)
    through the collapse without recomputing it.
    """

    chunks: list[ChunkRecord]
    kept: list[int]
    stats: DuplicateStats


def duplicate_stats(chunks: list[ChunkRecord]) -> DuplicateStats:
    """Measure the exact-duplicate rate of `chunks` without changing them."""
    return _stats(_group_sizes(chunks))


def collapse_duplicate_chunks(chunks: list[ChunkRecord]) -> Collapse:
    """Keep the FIRST chunk of every distinct text; fold the rest into its occurrence metadata.

    Build order is the corpus's sorted document order, so the survivor -- and therefore the
    persisted store -- is deterministic. Input records are never mutated: a collapsed survivor is
    a shallow copy with a new `metadata`, and an uncollapsed chunk is passed through untouched, so
    a corpus with no duplicates keeps byte-identical chunk records.
    """
    first_by_text: dict[str, int] = {}
    survivors: list[ChunkRecord] = []
    kept: list[int] = []
    occurrences: dict[int, list[DuplicateOccurrence]] = {}
    for position, chunk in enumerate(chunks):
        index = first_by_text.get(chunk["text"])
        if index is None:
            first_by_text[chunk["text"]] = len(survivors)
            survivors.append(chunk)
            kept.append(position)
            continue
        occurrences.setdefault(index, []).append(_occurrence(chunk))
    for index, copies in occurrences.items():
        survivors[index] = _with_occurrences(survivors[index], copies)
    sizes = [len(occurrences.get(i, ())) + 1 for i in range(len(survivors))]
    return Collapse(chunks=survivors, kept=kept, stats=_stats(sizes))


def expand_duplicate_chunks(chunks: list[ChunkRecord]) -> tuple[list[ChunkRecord], list[int]]:
    """Undo collapse: every copy back as its own record, plus the row each copy was indexed under.

    A dropped copy is stored as its full record minus the (identical) text, so expansion is
    exact. Records come back grouped by document in ascending offset order -- the from-scratch
    build order -- so a caller can merge them per document; the second list maps each expanded
    record to the position of the survivor it came from, which is the embedding row every copy of
    that text shares.
    """
    expanded: list[tuple[ChunkRecord, int]] = []
    for row, chunk in enumerate(chunks):
        copies = duplicate_occurrences(chunk)
        if not copies:
            expanded.append((chunk, row))
            continue
        expanded.append((_without_occurrences(chunk), row))
        expanded.extend((_restored(copy, chunk["text"]), row) for copy in copies)
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
    measured = (
        f"duplicates: {stats['duplicate_chunks']}/{stats['n']} chunks "
        f"({stats['duplicate_share']:.1%}) byte-identical to another, "
        f"{stats['groups']} groups, largest {stats['largest_group']} copies"
    )
    if collapsed:
        return f"{measured} -> {stats['unique']} indexed ({stats['collapsed']} collapsed)"
    return (
        f"{measured} -> all {stats['n']} indexed (--keep-duplicate-chunks; "
        f"identical text ties exactly, broken by chunk_id)"
    )


def _occurrence(chunk: ChunkRecord) -> DuplicateOccurrence:
    """The dropped chunk's whole record minus its text (identical to the survivor's by design)."""
    return cast(DuplicateOccurrence, {key: value for key, value in chunk.items() if key != "text"})


def _restored(occurrence: DuplicateOccurrence, text: str) -> ChunkRecord:
    return cast(ChunkRecord, {**occurrence, "text": text})


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


def _group_sizes(chunks: list[ChunkRecord]) -> list[int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk["text"]] = counts.get(chunk["text"], 0) + 1
    return list(counts.values())


def _stats(group_sizes: list[int]) -> DuplicateStats:
    n = sum(group_sizes)
    unique = len(group_sizes)
    repeated = [size for size in group_sizes if size > 1]
    duplicate_chunks = sum(repeated)
    return {
        "n": n,
        "unique": unique,
        "collapsed": n - unique,
        "duplicate_chunks": duplicate_chunks,
        "duplicate_share": duplicate_chunks / n if n else 0.0,
        "groups": len(repeated),
        "largest_group": max(group_sizes, default=1) if group_sizes else 1,
    }
