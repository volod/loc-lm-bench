"""Published-cell geometry and design loading for context-policy audits."""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
)
from llb.bench.agentic_policy_change_audit import (
    AUDITABLE_FIELDS,
    AUDITED_DESIGN_PATHS,
    CAP_FITTING_KINDS,
    GEOMETRY_KINDS,
    KIND_COLLAPSE,
    KIND_CONSTANT_SWEEP,
    KIND_FOLD_STEP,
    KIND_HARNESS,
    KIND_INTERACTION,
    KIND_KEEP_LONG,
    KIND_SURFACE,
    KIND_TWO_FOLD,
)

# Cap-fitting cells publish a compact-minus-cap delta, so both arms are always replayed.
CAP_FITTING_POLICIES = (POLICY_OBSERVATION_CAP, POLICY_COMPACT)
# Studies whose cells sit flat at the design root (no surface / families / ladders nesting).
FLAT_CELL_KINDS = frozenset(
    {KIND_INTERACTION, KIND_CONSTANT_SWEEP, KIND_KEEP_LONG, KIND_HARNESS, KIND_TWO_FOLD}
)


def declared_geometry(design: dict[str, object], study_kind: str) -> list[dict[str, object]]:
    """Flatten one study's declared cells into the shared replay geometry."""
    held = held_fixed(design, study_kind)
    default_share = held.get("compact_share")
    groups: list[tuple[int | None, dict[str, object]]]
    if study_kind == KIND_SURFACE:
        groups = [(None, cast(dict[str, object], design["surface"]))]
    elif study_kind in FLAT_CELL_KINDS:
        groups = [(None, design)]
    elif study_kind == KIND_COLLAPSE:
        groups = [
            (int(cast(int, family["depth"])), family)
            for family in cast(list[dict[str, object]], design["families"])
        ]
    elif study_kind == KIND_FOLD_STEP:
        groups = [
            (int(cast(int, ladder["depth"])), step)
            for ladder in cast(list[dict[str, object]], design["ladders"])
            for step in cast(list[dict[str, object]], ladder["steps"])
        ]
    else:
        raise ValueError(f"{study_kind!r} is not a readable geometry kind")
    return [
        _cell_geometry(
            cell, depth=depth, held=held, default_share=default_share, study_kind=study_kind
        )
        for depth, group in groups
        for cell in cast(list[dict[str, object]], group["cells"])
    ]


def load_audited_design(path: Path | str) -> dict[str, object]:
    from llb.bench.agentic_memory_transfer import load_transfer_design

    return load_transfer_design(path)


def load_audited_designs() -> dict[str, dict[str, object]]:
    from llb.core.paths import PROJECT_ROOT

    return {
        kind: load_audited_design(PROJECT_ROOT / path)
        for kind, path in AUDITED_DESIGN_PATHS.items()
    }


def _cell_geometry(
    cell: dict[str, object],
    *,
    depth: int | None,
    held: dict[str, object],
    default_share: object,
    study_kind: str,
) -> dict[str, object]:
    """One cell's replay geometry, including which policy arms and which fields it pins."""
    if "depth" in cell:
        resolved_depth = int(cast(int, cell["depth"]))
    elif depth is not None:
        resolved_depth = depth
    else:
        resolved_depth = int(cast(int, held["depth"]))
    return {
        "cell_id": cast(str, cell["cell_id"]),
        "depth": resolved_depth,
        "compact_share": _share(cell, default_share),
        "max_prompt_chars": int(cast(int, cell["max_prompt_chars"])),
        "pinned_fields": [name for name in AUDITABLE_FIELDS if name in cell],
        "policies": _policies(cell, study_kind),
    }


def _policies(cell: dict[str, object], study_kind: str) -> list[str]:
    """Which policy arms a cell was measured under -- cap-fitting always replays both."""
    if study_kind in CAP_FITTING_KINDS or study_kind == KIND_INTERACTION:
        return list(CAP_FITTING_POLICIES)
    raw = cell.get("policies")
    if not isinstance(raw, list) or not raw or not all(isinstance(name, str) for name in raw):
        raise ValueError(
            f"cell {cell.get('cell_id')!r} of {study_kind!r} must declare a non-empty policies list"
        )
    return list(cast(list[str], raw))


def _share(cell: dict[str, object], default: object) -> float:
    share = cell.get("compact_share", default)
    if share is None:
        raise ValueError(f"cell {cell.get('cell_id')!r} states no compact_share and none is held")
    return float(cast(float, share))


def held_fixed(design: dict[str, object], study_kind: str) -> dict[str, object]:
    if study_kind not in GEOMETRY_KINDS:
        raise ValueError(
            f"{study_kind!r} is not a readable geometry kind; choose from {GEOMETRY_KINDS}"
        )
    return cast(dict[str, object], design["held_fixed"])
