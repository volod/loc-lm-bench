"""Contracts for repeated-block rewrites and offset edits."""

from dataclasses import dataclass, field
from typing import NamedTuple
from typing_extensions import TypedDict


class TextEdit(NamedTuple):
    """One rewrite of `[start, end)` into `replacement`, in ascending non-overlapping order.

    `moved_to` is set only for a DROPPED repeat: it is the start offset (in the pre-strip text) of
    the surviving copy of the same text, which is where an offset inside the dropped block still
    resolves after the rewrite.
    """

    start: int
    end: int
    replacement: str
    moved_to: int | None = None


class RepeatCensus(TypedDict):
    """Block-level repetition of ONE document, measured before any rewrite."""

    blocks: int  # blank-line separated blocks in the document
    repeated_blocks: int  # blocks whose text occurs at least `min_repeats` times
    groups: int  # distinct texts repeating at least `min_repeats` times
    largest_group: int  # occurrences of the most repeated text (0 when nothing repeats)
    handled_groups: int  # of those groups, the ones a mode is allowed to rewrite
    handled_blocks: int  # blocks the mode actually rewrote


@dataclass(frozen=True)
class RepeatRewrite:
    """A rewritten document plus the edits that produced it and the census that motivated them."""

    text: str
    census: RepeatCensus
    edits: list[TextEdit] = field(default_factory=list)


class StrippedDoc(TypedDict):
    """A document's rewrite state: the stripped text and the edits that produced it.

    The citation and goldset remaps both read this to move offsets onto the stripped document.
    """

    text: str
    edits: list[TextEdit]
