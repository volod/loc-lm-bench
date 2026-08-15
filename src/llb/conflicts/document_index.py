"""Document ids written ONCE per bundle, and referenced by position everywhere else in it.

The record is linear in DOCUMENTS by design, but a document id is written down five times over: in
`documents`, in `chunks.stored`, in `chunks.comparable`, in each of the three `exclusions` maps, and
twice per row of the ranked `candidates` list. On a 250-document corpus that repetition is most of
the record, and it is the one part of it carrying no information -- `documents` is already the
corpus-order index every other key could name a position in.

So every key outside `documents` is that POSITION, as a decimal string (the maps) or an integer (the
candidate rows). The saving is the difference between an id and its ordinal, which grows with the id
and with how many maps mention it; what stays is exactly one copy of each id.

Two facts the interning has to survive:

- A store can be one build ahead of the corpus the run audited, so a chunk can carry a doc_id that
  the audited `documents` list never had. Such an id is appended to `extra_document_ids` and
  referenced by position like any other -- the alternative is a key that is sometimes an ordinal and
  sometimes an id, which no reader can tell apart.
- Bundles written before the interning key on the id itself. `DocumentNaming` is that seam: the
  schema version picks which form a record is read under, and both forms resolve to the same
  document ids, so every reading replays identically through either.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llb.core.contracts.common import JsonObject

# Where a doc_id goes when it is not one of the audited documents -- a store built from a corpus the
# run did not read. Absent on every bundle where the store and the corpus agree, which is all of
# them in normal operation, so the common record pays nothing for the case.
EXTRA_IDS_KEY = "extra_document_ids"


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
        self._corpus_size = len(doc_ids)
        self._extras: list[str] = []

    def position(self, doc_id: str) -> int:
        """`doc_id`'s index in this record's id table, appending it when the corpus had no such id."""
        position = self._positions.get(doc_id)
        if position is None:
            position = self._corpus_size + len(self._extras)
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

    @classmethod
    def by_id(cls) -> "DocumentNaming":
        """The pre-interning form, where every recorded key IS the document id."""
        return cls()

    @classmethod
    def by_position(cls, doc_ids: Sequence[str], extras: Sequence[str] = ()) -> "DocumentNaming":
        """The interned form: corpus order first, then whatever ids the corpus did not carry."""
        return cls(ids=(*doc_ids, *extras), interned=True)

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


def interned_counts(counts: Mapping[str, int], interner: DocumentInterner) -> JsonObject:
    """A `{doc_id: count}` map as the record carries it: keyed by corpus position."""
    return {interner.key(doc_id): int(count) for doc_id, count in counts.items()}


def named_counts(value: object, naming: DocumentNaming) -> dict[str, int]:
    """A recorded count map read back by document id, keeping only the entries that are a count."""
    if not isinstance(value, Mapping):
        return {}
    return {
        naming.name(str(key)): int(count) for key, count in value.items() if isinstance(count, int)
    }
