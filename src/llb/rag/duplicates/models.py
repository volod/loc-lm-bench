"""Contracts for duplicate collapse and census operations."""

from typing import NamedTuple
from typing_extensions import NotRequired, TypedDict
from llb.core.contracts.rag import ChunkRecord


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
