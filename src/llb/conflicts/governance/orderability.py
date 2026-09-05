"""Count orderable document pairs from governance keys without enumerating pairs."""

from collections import Counter
from collections.abc import Hashable, Sequence

from llb.conflicts.governance.editions import ORDERING_FIELDS, edition_key
from llb.core.contracts.common import JsonObject


def unordered_pairs(count: int) -> int:
    """How many unordered pairs `count` items make."""
    return count * (count - 1) // 2


def _same_key_pairs(keys: Sequence[Hashable]) -> int:
    """Pairs sharing a key, counted from the key multiset rather than from the pairs."""
    return sum(unordered_pairs(count) for count in Counter(keys).values())


def _differing_key_pairs(keys: Sequence[Hashable]) -> int:
    """Pairs whose keys differ. Every key here exists; absent keys are filtered first."""
    return unordered_pairs(len(keys)) - _same_key_pairs(keys)


def _both_differing_pairs(keyed: Sequence[tuple[Hashable, Hashable]]) -> int:
    """Pairs differing in both keys -- the overlap two per-field counts claim."""
    return (
        unordered_pairs(len(keyed))
        - _same_key_pairs([first for first, _ in keyed])
        - _same_key_pairs([second for _, second in keyed])
        + _same_key_pairs(keyed)
    )


def orderable_pair_count(document_governance: Sequence[JsonObject]) -> int:
    """Count pairs `compare_editions` can order in linear time.

    A pair is orderable on a field when both keys exist and differ. With the two supported
    ordering fields, inclusion-exclusion over their key multisets gives the exact union without
    visiting the pair space. The explicit unpack makes a third field fail rather than silently
    under-count.
    """
    date_field, version_field = ORDERING_FIELDS
    keyed = [
        (edition_key(governance, date_field), edition_key(governance, version_field))
        for governance in document_governance
    ]
    return (
        _differing_key_pairs([date for date, _ in keyed if date is not None])
        + _differing_key_pairs([version for _, version in keyed if version is not None])
        - _both_differing_pairs([keys for keys in keyed if None not in keys])
    )
