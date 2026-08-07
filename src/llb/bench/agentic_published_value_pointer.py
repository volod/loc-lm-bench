"""Address one field of a run aggregate.

The field pointer is a dotted path with one extra form, a row selector, because every one of these
aggregates keys its per-depth rows by a field rather than by position:

    depth_surface[depth=6].crossover_max_prompt_chars
    depth_ladders[depth=10].boundary.guard_boundary_chars
    cap_peak_prompt_chars.6

One walk serves both sources the resolution reads, because both are the same bytes: the repo's
committed copy of the aggregate, and the run artifact under DATA_DIR on a host that still has it.
"""

import re

# `name[key=value]`: pick the row of the `name` list whose integer `key` field is `value`.
_ROW_SELECTOR = re.compile(r"^(?P<name>\w+)\[(?P<key>\w+)=(?P<value>-?\d+)\]$")


def read_field(payload: dict[str, object], field: str, *, where: str) -> object:
    """Walk a field pointer, naming the segment that failed rather than the whole path."""
    node: object = payload
    for segment in field.split("."):
        node = _step(node, segment, field=field, where=where)
    return node


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
