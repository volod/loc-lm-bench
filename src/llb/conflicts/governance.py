"""Order two claim sides by edition, using only the recorded governance fields.

Staleness is orthogonal to the relation: a duplicate pair can be dated and a contradiction need
not be. Keeping it separate is what lets the claim tier promote a dated contradiction to
`superseded_by` without the model ever seeing a date, and lets an undated contradiction stay an
honest `contradicts` that a human has to settle.
"""

import re
from collections.abc import Callable
from typing import Any

from llb.conflicts.models import Staleness
from llb.core.contracts.common import JsonObject

_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_VERSION_PART = re.compile(r"\d+|[A-Za-z]+")

SIDE_A = "a"
SIDE_B = "b"
BASIS_EFFECTIVE_DATE = "effective_date"
BASIS_VERSION = "version"
# The fields `compare_editions` can order on, named beside the function itself so the coverage
# counts, the stage attribution, and the ordering cannot part company over which fields those are.
ORDERING_FIELDS = (BASIS_EFFECTIVE_DATE, BASIS_VERSION)


def _date_key(value: str) -> tuple[int, int, int] | None:
    match = _DATE.match(value.strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _version_key(value: str) -> tuple[tuple[int, Any], ...]:
    """Split a version into comparable parts; numeric runs compare numerically."""
    parts: list[tuple[int, Any]] = [
        (0, int(token)) if token.isdigit() else (1, token.casefold())
        for token in _VERSION_PART.findall(value.strip())
    ]
    return tuple(parts)


_KEY_OF_BASIS: dict[str, Callable[[str], Any]] = {
    BASIS_EFFECTIVE_DATE: _date_key,
    BASIS_VERSION: _version_key,
}


def _compare(a_value: str | None, b_value: str | None, key: Callable[[str], Any]) -> str | None:
    """`"a"` / `"b"` for the greater side under `key`, or None when not orderable."""
    if not a_value or not b_value or a_value == b_value:
        return None
    a_key, b_key = key(a_value), key(b_value)
    if a_key is None or b_key is None or a_key == b_key:
        return None
    return SIDE_A if a_key > b_key else SIDE_B


def edition_key(governance: JsonObject, field: str) -> Any | None:
    """The key `compare_editions` orders `field` on, or None when this record cannot be ordered by it.

    Two records are orderable by `field` exactly when both keys exist and DIFFER -- the same three
    conditions `_compare` applies, restated in a form that can be grouped instead of enumerated, so
    a count over a corpus's document pairs stays linear in the documents rather than quadratic in
    the pairs. It is the same key function either way, so the count cannot drift from the ordering.
    """
    value = governance.get(field)
    if not isinstance(value, str) or not value:
        return None
    return _KEY_OF_BASIS[field](value)


def compare_editions(a: JsonObject, b: JsonObject) -> Staleness:
    """Which governance record is the newer edition; `effective_date` wins over `version`."""
    a_date = a.get("effective_date")
    b_date = b.get("effective_date")
    newer = _compare(
        a_date if isinstance(a_date, str) else None,
        b_date if isinstance(b_date, str) else None,
        _date_key,
    )
    if newer is not None:
        return Staleness(newer_side=newer, basis=BASIS_EFFECTIVE_DATE)
    a_version = a.get("version")
    b_version = b.get("version")
    newer = _compare(
        a_version if isinstance(a_version, str) else None,
        b_version if isinstance(b_version, str) else None,
        _version_key,
    )
    if newer is not None:
        return Staleness(newer_side=newer, basis=BASIS_VERSION)
    return Staleness()
