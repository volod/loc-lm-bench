"""One restated row per published crossover, in the FORM that crossover was published in.

Three forms, three different things to do. An interpolated GUARD is re-interpolated over the
substituted cells and placed back on the deterministic step ladder. A fold-step BOUNDARY is a
property of that ladder rather than of a measured cost, so it is confirmed directly by the study
that re-measured its cells. A portable RATIO is neither: it is a quotient the collapse derived from
the surface's interpolated guard, so its own eight cells being bound-invariant says nothing about it
-- the guard it divides is exactly the number the restatement moves, and it is restated here from
that guard over the same depth's re-measured cap peak.

The invariance criterion follows the form as well. A guard holds while it names the same fold step,
because that is where cost changes; the ratio names no step of its own and was published as a band
across the tested depths, so it holds while it stays inside that band at the precision it is quoted
to.
"""

from typing import cast

from llb.bench.agentic_memory_boundary_crossover import READING_BRACKETED
from llb.bench.agentic_memory_cap_audit import VERDICT_SENSITIVE
from llb.bench.agentic_memory_fold_step_ladder import (
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    foldable_fold_steps,
)
from llb.bench.agentic_memory_crossover_restatement_placement import (
    DERIVED_RATIO_SOURCE_KIND,
    prompt_sequence,
)
from llb.bench.agentic_memory_crossover_restatement_reading import (
    BASIS_ALREADY_MEASURED,
    BASIS_DERIVED,
    BASIS_INVARIANT,
    BASIS_RESTATED,
    CRITERION_BAND,
    CRITERION_FOLD_STEP,
    FORM_INTERPOLATED,
    FORM_PORTABLE_RATIO,
)


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
        "invariance_criterion": _criterion(cast(str, published["form"])),
        "invariance_holds": True,
        "basis": BASIS_INVARIANT if not sensitive else None,
    }
    restated = _restated_surface(surfaces, depth)
    if published["form"] == FORM_PORTABLE_RATIO:
        # The one form whose contributing cells are not the ones that restate it: the collapse's own
        # cells fix the ratio's PORTABILITY, the surface's guard fixes its VALUE.
        return row if restated is None else _portable_ratio_row(row, published, restated)
    if not sensitive:
        return row
    if published["form"] != FORM_INTERPOLATED:
        # A fold-step boundary is a property of the deterministic ladder, not of a measured cost: its
        # sensitive cells were re-measured by the summarize-input-cap study, which confirmed the
        # boundary directly. Nothing here can restate it more strongly.
        row["basis"] = BASIS_ALREADY_MEASURED
        return row
    if restated is None:
        return row
    return _interpolated_row(row, published, restated, designs[kind], depth)


def _criterion(form: str) -> str:
    return CRITERION_BAND if form == FORM_PORTABLE_RATIO else CRITERION_FOLD_STEP


def _restated_surface(surfaces: list[dict[str, object]], depth: int) -> dict[str, object] | None:
    """The re-interpolated surface at this depth, or None when nothing there is bracketed."""
    return next(
        (
            surface
            for surface in surfaces
            if int(cast(int, surface["depth"])) == depth and surface["reading"] == READING_BRACKETED
        ),
        None,
    )


def _portable_ratio_row(
    row: dict[str, object], published: dict[str, object], restated: dict[str, object]
) -> dict[str, object]:
    """Derive the portable ratio from the restated guard over the re-measured cap peak.

    Both inputs come from ONE restated surface row, which is what makes this a restatement rather
    than a recomputation in prose: the trigger is the runtime's own arithmetic on the re-interpolated
    guard, and the peak is the one measured off the same prompt sequence that guard was placed on.
    Dividing by the published peak instead would rescale the ratio whenever the task world moved,
    which is the mistake the cap-peak comparison row exists to make visible.
    """
    guard = float(cast(float, restated["crossover_max_prompt_chars"]))
    peak = int(cast(int, restated["cap_peak_prompt_chars"]))
    share = float(cast(float, published["compact_share"]))
    trigger = compaction_trigger_chars(int(guard), share)
    ratio = trigger / peak
    low, high = (float(edge) for edge in cast(list[float], published["published_band"]))
    row.update(
        {
            "restated_value": ratio,
            "restated_trigger_chars": trigger,
            "restated_cap_peak_prompt_chars": peak,
            "derived_from_study_kind": DERIVED_RATIO_SOURCE_KIND,
            "derived_from_guard_chars": guard,
            "basis": BASIS_DERIVED,
        }
    )
    quoted = round(ratio, int(cast(int, published["band_decimals"])))
    row["invariance_holds"] = low <= quoted <= high
    return row


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
    sequence = prompt_sequence(design, depth)
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
    row["invariance_holds"] = row["restated_fold_step"] == published["fold_step"]
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
