"""Re-derive a published aggregate field from the evidence beside it.

A committed aggregate and a digest over it make the published value reviewable, but the pair can
still be fabricated together. The next cheap constraint is INTERNAL: a resolved field must be the
number the same aggregate's cells and geometry produce. Each supported pointer therefore has one
derivation here, using the production arithmetic that originally built its aggregate row.
"""

from collections.abc import Mapping
import re
from typing import Never, cast

from llb.bench.agentic_memory_boundary_crossover import depth_surface_row
from llb.bench.agentic_memory_boundary_probe import cap_prompt_sequence
from llb.bench.agentic_memory_fold_step_ladder import measured_cap_peak
from llb.bench.agentic_memory_fold_step_rows import depth_fold_row, step_rows
from llb.bench.agentic_memory_trigger_collapse_reading import annotate_fold_steps
from llb.bench.agentic_published_value_pointer import read_field

_SURFACE_FIELD = re.compile(
    r"^depth_surface\[depth=(?P<depth>-?\d+)\]\.crossover_max_prompt_chars$"
)
_FOLD_BOUNDARY_FIELD = re.compile(
    r"^depth_ladders\[depth=(?P<depth>-?\d+)\]\.boundary\.guard_boundary_chars$"
)
_CAP_PEAK_FIELD = re.compile(r"^cap_peak_prompt_chars\.(?P<depth>-?\d+)$")


def validate_aggregate_field(
    payload: dict[str, object], field: str, resolved: float, *, where: str
) -> None:
    """Refuse a resolved field that the aggregate's own evidence does not reproduce exactly."""
    match = _SURFACE_FIELD.fullmatch(field)
    if match is not None:
        expected = _derive_surface(payload, int(match["depth"]), where=where)
    else:
        match = _FOLD_BOUNDARY_FIELD.fullmatch(field)
        if match is not None:
            expected = _derive_fold_boundary(payload, int(match["depth"]), where=where)
        else:
            match = _CAP_PEAK_FIELD.fullmatch(field)
            if match is None:
                raise ValueError(
                    f"{where}: the provenance field {field!r} has no registered aggregate-internal "
                    "derivation, so the committed bytes can state it without their own cells or "
                    "geometry producing it"
                )
            expected = float(_derive_cap_peak(payload, int(match["depth"]), where=where))
    if expected != resolved:
        raise ValueError(
            f"{where}: the aggregate records {resolved!r} at {field!r}, while its own cells and "
            f"geometry derive {expected!r} -- the committed aggregate is internally inconsistent"
        )


def _derive_surface(payload: dict[str, object], depth: int, *, where: str) -> float:
    """Re-run the published linear interpolation over this aggregate's scored cells."""
    cells = [cell for cell in _rows(payload, "cells", where=where) if cell.get("depth") == depth]
    if not cells:
        _cannot_derive(where, f"the surface records no depth {depth} cells")
    try:
        row = depth_surface_row(
            depth,
            cells,
            cap_peak_prompt_chars=_derive_cap_peak(payload, depth, where=where),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _cannot_derive(where, f"the depth {depth} surface cells are unreadable: {exc}")
    return _derived_number(row.get("crossover_max_prompt_chars"), where=where, field="surface")


def _derive_fold_boundary(payload: dict[str, object], depth: int, *, where: str) -> float:
    """Rebuild the requested ladder from its cells, prompt sequence, share, and step rule."""
    ladder = _mapping(
        read_field(payload, f"depth_ladders[depth={depth}]", where=where),
        where=where,
        name=f"depth {depth} ladder",
    )
    sequence = _integer_sequence(ladder.get("cap_prompt_sequence"), where=where, depth=depth)
    share = _number(ladder.get("compact_share"), where=where, name="compact_share")
    rule = _mapping(payload.get("step_rule"), where=where, name="step_rule")
    fraction = _number(
        rule.get("within_step_cap_cost_fraction"),
        where=where,
        name="step_rule.within_step_cap_cost_fraction",
    )
    cells = [cell for cell in _rows(payload, "cells", where=where) if cell.get("depth") == depth]
    if not cells:
        _cannot_derive(where, f"the fold-step aggregate records no depth {depth} cells")
    try:
        annotated = annotate_fold_steps(cells, {depth: sequence})
        steps = step_rows(
            annotated,
            prompt_sequence=sequence,
            compact_share=share,
            cap_cost_fraction=fraction,
        )
        rebuilt = depth_fold_row(
            depth,
            steps,
            prompt_sequence=sequence,
            compact_share=share,
            cap_peak_prompt_chars=measured_cap_peak(
                sequence, geometry=f"the published depth {depth} fold-step ladder"
            ),
            reference_guard=None,
        )
        boundary = _mapping(rebuilt.get("boundary"), where=where, name="rebuilt boundary")
    except (KeyError, TypeError, ValueError) as exc:
        _cannot_derive(where, f"the depth {depth} fold-step cells are unreadable: {exc}")
    return _derived_number(boundary.get("guard_boundary_chars"), where=where, field="boundary")


def _derive_cap_peak(payload: dict[str, object], depth: int, *, where: str) -> int:
    """Recreate the prompt sequence from the aggregate's recorded geometry, then take its peak."""
    cells = _rows(payload, "cells", where=where)
    if not any(cell.get("depth") == depth for cell in cells):
        _cannot_derive(where, f"the aggregate records no depth {depth} cell for this cap peak")
    held = _mapping(payload.get("held_fixed"), where=where, name="held_fixed")
    try:
        sequence = cap_prompt_sequence(
            depth=depth,
            n_tasks=_integer(held.get("n_tasks"), where=where, name="held_fixed.n_tasks"),
            pad_chars=_integer(held.get("pad_chars"), where=where, name="held_fixed.pad_chars"),
            max_steps_margin=_integer(
                held.get("max_steps_margin"), where=where, name="held_fixed.max_steps_margin"
            ),
            observation_cap_chars=_integer(
                held.get("observation_cap_chars"),
                where=where,
                name="held_fixed.observation_cap_chars",
            ),
            observation_head_share=_number(
                held.get("observation_head_share"),
                where=where,
                name="held_fixed.observation_head_share",
            ),
        )
        return measured_cap_peak(sequence, geometry=f"the published depth {depth} cap geometry")
    except (KeyError, TypeError, ValueError) as exc:
        _cannot_derive(where, f"the depth {depth} cap prompt sequence is unreadable: {exc}")


def _rows(payload: dict[str, object], name: str, *, where: str) -> list[dict[str, object]]:
    value = payload.get(name)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        _cannot_derive(where, f"`{name}` must be a list of aggregate rows")
    return cast(list[dict[str, object]], value)


def _mapping(value: object, *, where: str, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _cannot_derive(where, f"`{name}` must be an aggregate object")
    return dict(cast(Mapping[str, object], value))


def _integer_sequence(value: object, *, where: str, depth: int) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        _cannot_derive(where, f"the depth {depth} `cap_prompt_sequence` must contain integers")
    return cast(list[int], value)


def _integer(value: object, *, where: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _cannot_derive(where, f"`{name}` must be an integer, got {value!r}")
    return cast(int, value)


def _number(value: object, *, where: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _cannot_derive(where, f"`{name}` must be numeric, got {value!r}")
    return float(cast(float, value))


def _derived_number(value: object, *, where: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _cannot_derive(where, f"the aggregate's cells produce no numeric {field}, got {value!r}")
    return float(cast(float, value))


def _cannot_derive(where: str, reason: str) -> Never:
    raise ValueError(
        f"{where}: the published aggregate cannot re-derive its resolved field: {reason}"
    )
