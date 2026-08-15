"""The head and tail every document id shares, written once instead of once per document.

With every key outside `documents` naming a corpus POSITION (`document_index.py`) and a document
with nothing to order on recorded as the bare id (`bundle_record.py`), the id table is the record's
remaining cost and it is nothing but ids. So the only lever left is on the id STRING itself -- and a
corpus-relative id is a PATH, which usually means one directory and one file type repeated per
document: 250 documents named `squad/<hex>.txt` write `squad/` and `.txt` 250 times each for no
information at all.

So the record writes that head and tail ONCE and each entry keeps only its stem. The join is exact
by construction -- `prefix + stem + suffix` is the id it was folded from, character for character --
which matters more than the bytes it saves: an id is the join key for `findings.jsonl` and for the
store, so a lossy round-trip would be worse than the repetition it removes.

The fold is applied only where it PAYS FOR ITSELF. A corpus whose documents sit in many directories
under mixed extensions shares nothing to fold, and a corpus of seven documents does not repeat its
`.md` often enough to earn the two keys the fold costs -- in both cases the affix is empty and the
entries are ids, byte for byte as they were before. That gate is also what makes the form
self-describing: the keys are present exactly when the entries are stems, so no reader needs
`schema_version` to tell a stem from an id.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llb.conflicts.record_fold import json_bytes
from llb.core.contracts.common import JsonObject

PREFIX_KEY = "document_id_prefix"
SUFFIX_KEY = "document_id_suffix"

# What one recorded key costs beyond its own value: the two quotes around the name, the colon, and
# the comma separating it from the next key.
KEY_OVERHEAD_BYTES = 4


def _common_prefix(values: Sequence[str]) -> str:
    """The longest head every value shares. Only the extremes can disagree, so only they are read."""
    if not values:
        return ""
    low, high = min(values), max(values)
    length = 0
    while length < len(low) and length < len(high) and low[length] == high[length]:
        length += 1
    return low[:length]


@dataclass(frozen=True)
class IdAffix:
    """The fold one record's document ids were written under, or the empty fold.

    The empty fold is not a degenerate case to guard against downstream: `stem` and `expand` are
    both the identity under it, so a reading is written once and works at either form.
    """

    prefix: str = ""
    suffix: str = ""

    @classmethod
    def over(cls, doc_ids: Sequence[str]) -> "IdAffix":
        """The fold these ids support, or the empty fold when it would not pay for itself.

        The suffix is taken over what the PREFIX left behind, so the two can never overlap: ids
        `["ab", "abc"]` fold to the prefix `ab` and no suffix, rather than to two affixes that
        between them are longer than the id they came from.
        """
        prefix = _common_prefix(doc_ids)
        stems = [doc_id[len(prefix) :] for doc_id in doc_ids]
        suffix = _common_prefix([stem[::-1] for stem in stems])[::-1]
        folded = cls(prefix=prefix, suffix=suffix)
        return folded if folded.pays_for_itself(len(doc_ids)) else cls()

    def pays_for_itself(self, documents: int) -> bool:
        """Whether folding `documents` ids saves more bytes than recording the fold costs.

        This is this fold's half of the rule in `record_fold.py`. A corpus that shares no head or
        tail folds to nothing by arithmetic, and a corpus too small to amortize the two keys is
        refused the fold rather than charged for it -- which is the case on every bundle here below
        about a dozen documents. The saving is counted rather than measured against a built table,
        because the table cannot be built until the affix is decided.
        """
        shared = len(self.prefix.encode("utf-8")) + len(self.suffix.encode("utf-8"))
        if not shared:
            return False
        cost = sum(
            len(key) + json_bytes(value) + KEY_OVERHEAD_BYTES
            for key, value in self.payload().items()
        )
        return documents * shared > cost

    def payload(self) -> JsonObject:
        """The keys the record carries, each absent when it has nothing to say."""
        return {
            key: value
            for key, value in ((PREFIX_KEY, self.prefix), (SUFFIX_KEY, self.suffix))
            if value
        }

    def stem(self, doc_id: str) -> str:
        """The part of `doc_id` the record actually writes down."""
        return doc_id[len(self.prefix) : len(doc_id) - len(self.suffix)]

    def expand(self, stem: str) -> str:
        """The id a recorded stem was folded from, character for character."""
        return f"{self.prefix}{stem}{self.suffix}"

    @classmethod
    def from_record(cls, record: JsonObject) -> "IdAffix":
        """The fold a bundle was written under -- the empty fold on every bundle that folded none."""
        return cls(
            prefix=prefix if isinstance(prefix := record.get(PREFIX_KEY), str) else "",
            suffix=suffix if isinstance(suffix := record.get(SUFFIX_KEY), str) else "",
        )
