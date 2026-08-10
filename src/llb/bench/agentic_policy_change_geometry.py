"""Published-cell geometry and design loading for context-policy audits."""

from pathlib import Path
from typing import cast

from llb.bench.agentic_policy_change_audit import (
    AUDITABLE_FIELDS,
    AUDITED_DESIGN_PATHS,
    GEOMETRY_KINDS,
    KIND_COLLAPSE,
    KIND_INTERACTION,
    KIND_SURFACE,
)


def declared_geometry(design: dict[str, object], study_kind: str) -> list[dict[str, object]]:
    """Flatten one study's declared cells into the shared replay geometry."""
    held = held_fixed(design, study_kind)
    default_share = held.get("compact_share")
    groups: list[tuple[int | None, dict[str, object]]]
    if study_kind == KIND_SURFACE:
        groups = [(None, cast(dict[str, object], design["surface"]))]
    elif study_kind == KIND_INTERACTION:
        groups = [(None, design)]
    elif study_kind == KIND_COLLAPSE:
        groups = [
            (int(cast(int, family["depth"])), family)
            for family in cast(list[dict[str, object]], design["families"])
        ]
    else:
        groups = [
            (int(cast(int, ladder["depth"])), step)
            for ladder in cast(list[dict[str, object]], design["ladders"])
            for step in cast(list[dict[str, object]], ladder["steps"])
        ]
    return [
        {
            "cell_id": cast(str, cell["cell_id"]),
            "depth": int(cast(int, cell.get("depth", depth))),
            "compact_share": _share(cell, default_share),
            "max_prompt_chars": int(cast(int, cell["max_prompt_chars"])),
            "pinned_fields": [name for name in AUDITABLE_FIELDS if name in cell],
        }
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
