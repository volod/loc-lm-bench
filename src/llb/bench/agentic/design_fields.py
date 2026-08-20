"""Typed reads of one JSON design field, so a validator reads as the contract it is checking.

A design arrives as `dict[str, object]` -- the honest type for parsed JSON -- and every check over it
used to spend a line on the cast before it could spend one on the rule:
`int(cast(int, matrix.get("n_tasks", 0))) < 6`. The cast is noise in the place where the CONTRACT is
supposed to be readable, and it is repeated once per field across every study's validator.

These readers do the coercion once, named. A missing field takes the caller's default (a design that
omits a field fails the rule, not the read), and a present field is coerced the way the old inline
`int(...)` / `float(...)` / `str(...)` did -- so a value of the wrong JSON type still raises where it
always did, rather than being silently defaulted into a passing number.
"""

from collections.abc import Mapping
from typing import cast


def as_mapping(source: Mapping[str, object], key: str) -> dict[str, object]:
    """One nested object, empty when absent -- an absent block fails its own rules, not this read."""
    value = source.get(key)
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def as_rows(source: Mapping[str, object], key: str) -> list[dict[str, object]]:
    """One array of objects (a roster, a cell list), empty when absent."""
    value = source.get(key)
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], row) for row in value if isinstance(row, dict)]


def as_list(source: Mapping[str, object], key: str) -> list[object]:
    """One array of scalars, empty when absent."""
    value = source.get(key)
    return list(cast(list[object], value)) if isinstance(value, list) else []


def as_ints(source: Mapping[str, object], key: str) -> list[int]:
    """One array of integers (depths, seeds, task counts)."""
    return [int(cast(int, item)) for item in as_list(source, key)]


def as_floats(source: Mapping[str, object], key: str) -> list[float]:
    """One array of floats (compact shares, thresholds)."""
    return [float(cast(float, item)) for item in as_list(source, key)]


def as_strs(source: Mapping[str, object], key: str) -> list[str]:
    """One array of strings (placements, families, forbidden terms)."""
    return [str(item) for item in as_list(source, key)]


def as_int(source: Mapping[str, object], key: str, default: int = 0) -> int:
    """One integer field."""
    value = source.get(key)
    return default if value is None else int(cast(int, value))


def as_float(source: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """One float field."""
    value = source.get(key)
    return default if value is None else float(cast(float, value))


def as_str(source: Mapping[str, object], key: str, default: str = "") -> str:
    """One string field."""
    value = source.get(key)
    return default if value is None else str(value)


def as_bool(source: Mapping[str, object], key: str, *, default: bool = False) -> bool:
    """One boolean field, read strictly: only a JSON `true` is true.

    Strict because these designs use a THREE-valued convention -- `true`, `false`, and absent mean
    three different things to a cell (`require_cap_fits` is the shipped example) -- so coercing a
    missing field through `bool()` would turn "the design did not say" into "the design said no".
    """
    value = source.get(key)
    return default if value is None else value is True
