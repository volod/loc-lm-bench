"""Address one field of a run aggregate, and cut the slice of it that field needs.

The field pointer is a dotted path with one extra form, a row selector, because every one of these
aggregates keys its per-depth rows by a field rather than by position:

    depth_surface[depth=6].crossover_max_prompt_chars
    depth_ladders[depth=10].boundary.guard_boundary_chars
    cap_peak_prompt_chars.6

Reading and cutting live together because they are the same walk: the committed slice keeps the
artifact's shape, so a pointer that resolves against the artifact but not against its slice (or the
reverse) is impossible.
"""

import re
from typing import cast

# `name[key=value]`: pick the row of the `name` list whose integer `key` field is `value`.
_ROW_SELECTOR = re.compile(r"^(?P<name>\w+)\[(?P<key>\w+)=(?P<value>-?\d+)\]$")


def read_field(payload: dict[str, object], field: str, *, where: str) -> object:
    """Walk a field pointer, naming the segment that failed rather than the whole path."""
    node: object = payload
    for segment in field.split("."):
        node = _step(node, segment, field=field, where=where)
    return node


def merge_field_slice(target: dict[str, object], payload: dict[str, object], field: str) -> None:
    """Copy just the addressed path of `payload` into `target`, preserving the artifact's shape.

    Shape-preserving is the point: the committed slice is then read by the SAME pointer walk as the
    artifact, so a slice that resolves and an artifact that does not (or the reverse) is impossible.
    """
    _merge(target, payload, field.split("."), field=field)


def _step(node: object, segment: str, *, field: str, where: str) -> object:
    selector = _ROW_SELECTOR.match(segment)
    if selector is None:
        if not isinstance(node, dict) or segment not in node:
            raise ValueError(
                f"{where}: the field pointer {field!r} reaches no {segment!r} in the artifact"
            )
        return node[segment]
    name, key, value = selector["name"], selector["key"], int(selector["value"])
    rows = node.get(name) if isinstance(node, dict) else None
    row = _select_row(rows, key, value)
    if row is None:
        raise ValueError(
            f"{where}: the field pointer {field!r} selects the {name!r} row with {key}={value}, "
            "which the artifact does not carry"
        )
    return row


def _select_row(rows: object, key: str, value: int) -> dict[str, object] | None:
    if not isinstance(rows, list):
        return None
    return next(
        (
            row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get(key), int) and row[key] == value
        ),
        None,
    )


def _merge(
    target: dict[str, object], payload: dict[str, object], segments: list[str], *, field: str
) -> None:
    segment, rest = segments[0], segments[1:]
    selector = _ROW_SELECTOR.match(segment)
    if selector is None:
        if segment not in payload:
            raise ValueError(f"the field pointer {field!r} reaches no {segment!r} in the artifact")
        if not rest:
            target[segment] = payload[segment]
            return
        nested = payload[segment]
        if not isinstance(nested, dict):
            raise ValueError(f"the field pointer {field!r} walks into a non-object at {segment!r}")
        _merge(cast(dict[str, object], target.setdefault(segment, {})), nested, rest, field=field)
        return
    if not rest:
        raise ValueError(f"the field pointer {field!r} ends on a row selector, not on a field")
    name, key, value = selector["name"], selector["key"], int(selector["value"])
    row = _select_row(payload.get(name), key, value)
    if row is None:
        raise ValueError(
            f"the field pointer {field!r} selects a {name!r} row the artifact does not carry"
        )
    bucket = cast(list[dict[str, object]], target.setdefault(name, []))
    kept = _select_row(bucket, key, value)
    if kept is None:
        kept = {key: value}
        bucket.append(kept)
    _merge(kept, row, rest, field=field)
