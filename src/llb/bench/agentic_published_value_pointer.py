"""Address one field of a run aggregate.

The field pointer is a dotted path with one extra form, a row selector, because every one of these
aggregates keys its per-depth rows by a field rather than by position:

    depth_surface[depth=6].crossover_max_prompt_chars
    depth_ladders[depth=10].boundary.guard_boundary_chars
    cells[cell_id=surface-d10-g23000].measured_side
    cap_peak_prompt_chars.6

One walk serves both sources the resolution reads, because both are the same bytes: the repo's
committed copy of the aggregate, and the run artifact under DATA_DIR on a host that still has it.
The selector accepts an integer or a string value, so a depth row and a cell_id row share the walk.
"""

import re

# `name[key=value]`: pick the row of the `name` list whose `key` field equals `value`. Integers keep
# their unquoted form (`depth=6`); everything else is a string key (`cell_id=surface-d10-g23000`).
_ROW_SELECTOR = re.compile(r"^(?P<name>\w+)\[(?P<key>\w+)=(?P<value>[^\]]+)\]$")


def read_field(payload: dict[str, object], field: str, *, where: str) -> object:
    """Walk a field pointer, naming the segment that failed rather than the whole path."""
    node: object = payload
    for segment in _segments(field):
        node = _step(node, segment, field=field, where=where)
    return node


def _segments(field: str) -> list[str]:
    """Split on dots that are not inside a row selector.

    Cell ids carry dots (`collapse-d6-s0.4-g17500`), so a naive `field.split('.')` would break the
    selector open and look for a segment that never existed.
    """
    segments: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in field:
        if char == "[":
            depth += 1
            buf.append(char)
        elif char == "]":
            depth = max(0, depth - 1)
            buf.append(char)
        elif char == "." and depth == 0:
            segments.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    if buf:
        segments.append("".join(buf))
    return segments


def _step(node: object, segment: str, *, field: str, where: str) -> object:
    selector = _ROW_SELECTOR.match(segment)
    if selector is None:
        if not isinstance(node, dict) or segment not in node:
            raise ValueError(
                f"{where}: the field pointer {field!r} reaches no {segment!r} in the artifact"
            )
        return node[segment]
    name, key, raw = selector["name"], selector["key"], selector["value"]
    rows = node.get(name) if isinstance(node, dict) else None
    row = _select_row(rows, key, _selector_value(raw))
    if row is None:
        raise ValueError(
            f"{where}: the field pointer {field!r} selects the {name!r} row with {key}={raw}, "
            "which the artifact does not carry"
        )
    return row


def _selector_value(raw: str) -> int | str:
    """An integer when the token is one, otherwise the string the design wrote."""
    try:
        return int(raw)
    except ValueError:
        return raw


def _select_row(rows: object, key: str, value: int | str) -> dict[str, object] | None:
    if not isinstance(rows, list):
        return None
    return next(
        (row for row in rows if isinstance(row, dict) and row.get(key) == value),
        None,
    )
