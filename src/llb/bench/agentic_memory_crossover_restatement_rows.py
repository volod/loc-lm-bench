"""Row builders for the crossover restatement: cells, surfaces, cap peaks, and crossover rows.

Substituting a re-measured cell into a published grid and re-running that grid's own rule is the
whole restatement, so these builders reuse the published machinery rather than reimplementing it:
the surface's precondition gate (`surface_cell_row`) and its predeclared interpolation
(`depth_surface_row`). What is added is the check the restatement exists for -- whether the restated
number still names the fold step the published one named.

One rule governs where every number comes from: what the restated geometry can measure is measured
here off `_prompt_sequence` -- the guard ratio's cap peak as much as the fold step -- and what only
the published artifact holds is stated as its own comparison row (`cap_peak_rows`) rather than
divided into a measured one, because a quotient reports no disagreement between its parts.
"""

from typing import cast

from llb.bench.agentic_memory_boundary_crossover import READING_BRACKETED, depth_surface_row
from llb.bench.agentic_memory_boundary_probe import cap_prompt_sequence
from llb.bench.agentic_memory_fold_step_ladder import (
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    foldable_fold_steps,
    measured_cap_peak,
)
from llb.bench.agentic_memory_boundary_surface_cells import surface_cell_row
from llb.bench.agentic_memory_cap_audit import VERDICT_SENSITIVE
from llb.bench.agentic_policy_change_audit import KIND_SURFACE
from llb.bench.agentic_memory_crossover_restatement_reading import (
    BASIS_ALREADY_MEASURED,
    BASIS_INVARIANT,
    BASIS_RESTATED,
    FORM_INTERPOLATED,
    PEAK_INVARIANT,
    PEAK_MOVED,
    PEAK_UNPUBLISHED,
    REPORTING_CONFIDENCE,
)


def restated_cells(
    designs: dict[str, dict[str, object]], restated_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Gate the re-measured cells with the surface's own precondition contract."""
    held = cast(dict[str, object], designs[KIND_SURFACE]["held_fixed"])
    declared = {
        cast(str, cell["cell_id"]): cell
        for cell in cast(
            list[dict[str, object]],
            cast(dict[str, object], designs[KIND_SURFACE]["surface"])["cells"],
        )
    }
    return [
        surface_cell_row(
            row,
            declared[cast(str, row["restated_cell_id"])],
            held_fixed=held,
            confidence=REPORTING_CONFIDENCE,
        )
        for row in restated_rows
    ]


def restated_surfaces(
    designs: dict[str, dict[str, object]],
    published_surface: dict[str, object],
    cells: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Re-run the surface's published interpolation over its cells with the new ones substituted.

    The cap peak the guard ratio is divided by is MEASURED here, off the same prompt sequence the
    restated fold step is read from, rather than read out of the published aggregate. Mixing the two
    would state a freshly interpolated guard as a fraction of a peak a moved task world has retired:
    a changed `pad_chars`, observation cap, or step margin rescales the ratio with nothing in the run
    saying so, and the fold-step check notices only when the same drift also reshapes the step
    ladder, which a small move need not do.
    """
    design = designs[KIND_SURFACE]
    replacements = {cast(str, cell["cell_id"]): cell for cell in cells}
    substituted = [
        replacements.get(cast(str, cell["cell_id"]), cell)
        for cell in cast(list[dict[str, object]], published_surface["cells"])
    ]
    return [
        {
            **depth_surface_row(
                depth,
                [cell for cell in substituted if int(cast(int, cell["depth"])) == depth],
                cap_peak_prompt_chars=measured_cap_peak(
                    _prompt_sequence(design, depth),
                    geometry=f"the restated depth {depth} surface",
                ),
            ),
            "restated_cell_ids": sorted(
                cast(str, cell["cell_id"])
                for cell in cells
                if int(cast(int, cell["depth"])) == depth
            ),
        }
        for depth in sorted({int(cast(int, cell["depth"])) for cell in substituted})
    ]


def cap_peak_rows(
    published_surface: dict[str, object], surfaces: list[dict[str, object]]
) -> list[dict[str, object]]:
    """State the published cap peak beside the re-measured one, per restated depth.

    As a row the two are comparable, so a depth whose peak moved NAMES the move, and the ratio beside
    it is the one the geometry that measured the guard supports.
    """
    published = cast(dict[str, object], published_surface["cap_peak_prompt_chars"])
    return [
        _cap_peak_row(
            int(cast(int, surface["depth"])),
            published.get(str(cast(int, surface["depth"]))),
            int(cast(int, surface["cap_peak_prompt_chars"])),
            cast(float | None, surface["crossover_guard_ratio"]),
        )
        for surface in surfaces
    ]


def _cap_peak_row(
    depth: int, published: object | None, measured: int, ratio: float | None
) -> dict[str, object]:
    stated = None if published is None else int(cast(int, published))
    if stated is None:
        reading = PEAK_UNPUBLISHED
    else:
        reading = PEAK_INVARIANT if stated == measured else PEAK_MOVED
    return {
        "study_kind": KIND_SURFACE,
        "depth": depth,
        "published_cap_peak_prompt_chars": stated,
        "measured_cap_peak_prompt_chars": measured,
        "cap_peak_delta_chars": None if stated is None else measured - stated,
        "restated_guard_ratio": ratio,
        "reading": reading,
    }


def crossover_row(
    published: dict[str, object],
    designs: dict[str, dict[str, object]],
    audit: dict[str, object],
    surfaces: list[dict[str, object]],
) -> dict[str, object]:
    """One published crossover, restated where a run was needed and re-checked where it was not."""
    kind = cast(str, published["study_kind"])
    depth = int(cast(int, published["depth"]))
    sensitive = [
        row
        for row in cast(dict[str, list[dict[str, object]]], audit["per_study"])[kind]
        if row["verdict"] == VERDICT_SENSITIVE and int(cast(int, row["depth"])) == depth
    ]
    row: dict[str, object] = {
        **published,
        "n_bound_sensitive_cells": len(sensitive),
        "bound_sensitive_cell_ids": [cast(str, cell["cell_id"]) for cell in sensitive],
        "published_value": published.get("value"),
        "published_fold_step": published["fold_step"],
        "restated_value": None,
        "restated_fold_step": published["fold_step"],
        "names_same_fold_step": True,
        "basis": BASIS_INVARIANT if not sensitive else None,
    }
    if not sensitive:
        return row
    if published["form"] != FORM_INTERPOLATED:
        # A fold-step boundary is a property of the deterministic ladder, not of a measured cost: its
        # sensitive cells were re-measured by the summarize-input-cap study, which confirmed the
        # boundary directly. Nothing here can restate it more strongly.
        row["basis"] = BASIS_ALREADY_MEASURED
        return row
    restated = next(
        (surface for surface in surfaces if int(cast(int, surface["depth"])) == depth), None
    )
    if restated is None or restated["reading"] != READING_BRACKETED:
        return row
    return _interpolated_row(row, published, restated, designs[kind], depth)


def _interpolated_row(
    row: dict[str, object],
    published: dict[str, object],
    restated: dict[str, object],
    design: dict[str, object],
    depth: int,
) -> dict[str, object]:
    """Place the re-interpolated guard on the deterministic step ladder it has to stay inside."""
    guard = float(cast(float, restated["crossover_max_prompt_chars"]))
    share = float(cast(float, published["compact_share"]))
    sequence = _prompt_sequence(design, depth)
    row.update(
        {
            "restated_value": guard,
            "restated_bracket": restated["bracket"],
            "restated_guard_ratio": restated["crossover_guard_ratio"],
            "restated_fold_step": first_fold_step(
                sequence, compaction_trigger_chars(int(guard), share)
            ),
            "basis": BASIS_RESTATED,
            "fold_step_guard_interval": _published_guard_interval(
                sequence, published, depth, share
            ),
        }
    )
    row["names_same_fold_step"] = row["restated_fold_step"] == published["fold_step"]
    return row


def _published_guard_interval(
    sequence: list[int], published: dict[str, object], depth: int, share: float
) -> list[int]:
    """The published step's guard interval, or a refusal naming the PUBLISHED ROW it came from.

    Every other ladder caller reads a step it measured itself; this one reads a step off a COMMITTED
    artifact against a freshly measured sequence, so the two can disagree about the geometry without
    either being malformed. That disagreement is not an argument error -- it is exactly the drift the
    restatement exists to catch: the task world moved, and the fold step the published number was
    stated in is not one this geometry has any more. The interval arithmetic says that as a bare
    "outside an N-step sequence", which names neither the number nor the study that published it.

    The check is the FOLDABLE ladder rather than the sequence bounds, because a published crossover
    is a cost measured at a step an episode folded at: a step still in range but no longer reachable
    (an earlier prompt grew past it) has been retired by the geometry just as completely.
    """
    step = int(cast(int, published["fold_step"]))
    ladder = foldable_fold_steps(sequence)
    if step not in ladder:
        raise ValueError(
            f"the published {published['study_kind']} crossover at depth {depth} is stated at fold "
            f"step {step}, which the re-measured {len(sequence)}-step prompt sequence no longer has "
            f"(its foldable ladder is {ladder}): the task world moved under the published number, "
            "so there is no interval left to restate it inside"
        )
    return list(fold_step_guard_interval(sequence, step, share))


def _prompt_sequence(design: dict[str, object], depth: int) -> list[int]:
    held = cast(dict[str, object], design["held_fixed"])
    return cap_prompt_sequence(
        depth=depth,
        n_tasks=int(cast(int, held["n_tasks"])),
        pad_chars=int(cast(int, held["pad_chars"])),
        max_steps_margin=int(cast(int, held["max_steps_margin"])),
        observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
        observation_head_share=float(cast(float, held["observation_head_share"])),
    )
