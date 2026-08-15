"""Document ids written ONCE per bundle, and referenced by position everywhere else in it.

The record is linear in DOCUMENTS by design, but a document id is written down five times over: in
`documents`, in `chunks.stored`, in `chunks.comparable`, in each of the three `exclusions` maps, and
twice per row of the ranked `candidates` list. On a 250-document corpus that repetition is most of
the record, and it is the one part of it carrying no information -- `documents` is already the
corpus-order index every other key could name a position in.

So every key outside `documents` is that POSITION, as a decimal string (the maps) or an integer (the
candidate rows). The saving is the difference between an id and its ordinal, which grows with the id
and with how many maps mention it; what stays is exactly one copy of each id.

What is left in a map once its keys are ordinals is the COUNTS, and those repeat too: most documents
store one chunk and exclude none, so the same small number is written once per document. So a map
records the count most corpus documents share ONCE, under `default`, and lists only the documents
that differ (`interned_counts` / `named_counts`). Both folds are optional and gated the same way
(`record_fold.py`) -- a map where no count dominates keeps the plain form.

Two facts the interning has to survive:

- A store can be one build ahead of the corpus the run audited, so a chunk can carry a doc_id that
  the audited `documents` list never had. Such an id is appended to `extra_document_ids` and
  referenced by position like any other -- the alternative is a key that is sometimes an ordinal and
  sometimes an id, which no reader can tell apart.
- Bundles written before the interning key on the id itself. `DocumentNaming` is that seam: the
  schema version picks which form a record is read under, and both forms resolve to the same
  document ids, so every reading replays identically through either.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llb.conflicts.record_fold import smaller_form
from llb.core.contracts.common import JsonObject

# Where a doc_id goes when it is not one of the audited documents -- a store built from a corpus the
# run did not read. Absent on every bundle where the store and the corpus agree, which is all of
# them in normal operation, so the common record pays nothing for the case.
EXTRA_IDS_KEY = "extra_document_ids"

# The count most corpus documents share, written once instead of once per document. Safe as a
# literal key beside decimal positions because the fold only ever exists in an INTERNED record,
# where every other key of the map is a number.
COUNT_DEFAULT_KEY = "default"


class DocumentInterner:
    """Write side: the position a document id is recorded as, for one record.

    Mutable on purpose -- `position` is what discovers an id the corpus did not carry, and `extras`
    is only complete once every part of the record has been written.
    """

    def __init__(self, doc_ids: Sequence[str]) -> None:
        # First occurrence wins, so a corpus that lists one id twice still resolves that position
        # back to the same string.
        self._positions: dict[str, int] = {}
        for position, doc_id in enumerate(doc_ids):
            self._positions.setdefault(doc_id, position)
        self.corpus_size = len(doc_ids)
        self._extras: list[str] = []

    def position(self, doc_id: str) -> int:
        """`doc_id`'s index in this record's id table, appending it when the corpus had no such id."""
        position = self._positions.get(doc_id)
        if position is None:
            position = self.corpus_size + len(self._extras)
            self._extras.append(doc_id)
            self._positions[doc_id] = position
        return position

    def key(self, doc_id: str) -> str:
        """The same position as a JSON object key, which can only be a string."""
        return str(self.position(doc_id))

    @property
    def extras(self) -> list[str]:
        """The ids that were not corpus documents, in the order the record references them."""
        return list(self._extras)


@dataclass(frozen=True)
class DocumentNaming:
    """Read side: which document a recorded key names, at either form of the record."""

    ids: tuple[str, ...] = ()
    interned: bool = False
    # How many of `ids` are AUDITED corpus documents; the rest are the extras the store held. A
    # recorded default covers exactly the corpus documents, so the count is what separates them.
    corpus_size: int = 0

    @classmethod
    def by_id(cls) -> "DocumentNaming":
        """The pre-interning form, where every recorded key IS the document id."""
        return cls()

    @classmethod
    def by_position(cls, doc_ids: Sequence[str], extras: Sequence[str] = ()) -> "DocumentNaming":
        """The interned form: corpus order first, then whatever ids the corpus did not carry."""
        return cls(ids=(*doc_ids, *extras), interned=True, corpus_size=len(doc_ids))

    @property
    def corpus_ids(self) -> tuple[str, ...]:
        """The audited documents a recorded default speaks for, in corpus order."""
        return self.ids[: self.corpus_size]

    def name(self, key: str) -> str:
        """The document `key` refers to.

        A key that resolves to nothing -- out of range, not a number, or a table slot with no id --
        is returned as it stands rather than dropped: a record whose id table and maps disagree is
        already outside the contract, and the id-shaped reading of it is the one a reader can still
        act on.
        """
        if not self.interned:
            return key
        try:
            position = int(key)
        except ValueError:
            return key
        resolved = self.ids[position] if 0 <= position < len(self.ids) else ""
        return resolved or key


def _with_default(positions: JsonObject, corpus_size: int) -> JsonObject:
    """The same map with the count most corpus documents share written once, under `default`.

    A CORPUS document is the only thing a default speaks for. An id the audited corpus did not carry
    stays explicit: it is in the table because some map mentioned it, and the record cannot say that
    a document it never enumerated shares anything.

    A document absent from `positions` counts as 0 here, which is what absence already meant -- so
    the fold reads the map it is folding exactly as every consumer does.
    """
    corpus = [str(position) for position in range(corpus_size)]
    full = {key: int(positions.get(key, 0)) for key in corpus}
    modal, _ = Counter(full.values()).most_common(1)[0]
    return {
        COUNT_DEFAULT_KEY: modal,
        **{key: count for key, count in full.items() if count != modal},
        **{key: int(count) for key, count in positions.items() if key not in set(corpus)},
    }


def interned_counts(
    counts: Mapping[str, int], interner: DocumentInterner, *, absent_is_zero: bool = True
) -> JsonObject:
    """A `{doc_id: count}` map as the record carries it: keyed by corpus position.

    Most corpus documents usually share one count -- one stored chunk, no excluded chunk -- and on
    a corpus of thousands writing that count out per document is the map's whole cost. So it is
    written once and only the documents that differ are listed, WHEN that is smaller
    (`record_fold.py`): a map where every document differs, or a sparse map whose absent documents
    already say zero, keeps the plain form and pays nothing for the option.

    `absent_is_zero=False` declines the fold outright, for a map where a MISSING document is not the
    same claim as a zero one. The fold trades on those being the same -- it reads the map it folds
    the way every consumer reads it, and it writes a zero back as an absence -- so a map that
    distinguishes them (`recovery_floor`: no floor recovers this document, against a floor of 0)
    would lose that distinction rather than bytes.
    """
    positions = {interner.key(doc_id): int(count) for doc_id, count in counts.items()}
    if not absent_is_zero or not interner.corpus_size:
        return positions
    return smaller_form(_with_default(positions, interner.corpus_size), positions)


def named_counts(value: object, naming: DocumentNaming) -> dict[str, int]:
    """A recorded count map read back by document id, keeping only the entries that are a count.

    A recorded `default` is expanded over the corpus documents FIRST, so an explicit entry always
    wins over it, and the zeros are then dropped -- because in a folded map a zero is the absence
    the plain map expresses, and dropping it is what makes a folded map read back as exactly the
    mapping the plain one gives rather than as the same answers in a denser dict. A map with no
    default is returned as it stands, zeros included: only the writer knows whether absence meant
    zero there, and it declined the fold precisely where it did not.
    """
    if not isinstance(value, Mapping):
        return {}
    counts = {
        naming.name(str(key)): int(count)
        for key, count in value.items()
        if key != COUNT_DEFAULT_KEY and isinstance(count, int)
    }
    default = value.get(COUNT_DEFAULT_KEY)
    if not isinstance(default, int):
        return counts
    expanded = {**dict.fromkeys(naming.corpus_ids, default), **counts}
    return {doc_id: count for doc_id, count in expanded.items() if count}
