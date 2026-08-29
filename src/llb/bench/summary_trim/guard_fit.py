"""Fitting the middle-critical workload's prompt guard to the walk each family actually takes.

The middle-critical stratum only says anything where an episode REACHES its fold: both arms build
byte-identical prompts up to the transcript the fold offers, so a walk that ends before the trigger
is crossed ran no trim at all and leaves the stratum smaller than the design declared. Held at one
shared character constant, that is a per-family property nobody chose -- a family whose walk ends
early on a case simply never enters the regime under test, and the recovery rests on the cases that
happened to walk far enough.

So the guard is fitted per family, from that family's OWN measured walk, the way the repeated-fold
ladder fits its guard from the family's measured fold length. The measurement is a WALK CONTROL: the
same tasks under a guard whose model-free probe folds zero times, which is byte-identical to every
candidate guard's pre-fold prompts and therefore measures how far the family walks before any fold
could have changed the transcript.

What keeps the fit honest is that the band is filtered by the declared REGIME before it is scored.
A lower guard folds earlier, but it also folds a shorter transcript against a smaller summarize-input
bound, and below some guard the fold either elides nothing or the answer fact stops occupying the
stratum it was planted in. Those guards are refused with a reason rather than selected, so a fit
that cannot reach the declared stratum size reports which guard came closest and what stopped the
band there -- it never widens the band after the fact.
"""

from typing import cast

from llb.bench.agentic.context_policy import SUMMARY_TRIM_HEAD_TAIL
from llb.bench.agentic.design_fields import as_int, as_mapping, as_str
from llb.bench.context_policy.guard_band import (
    guard_grid,
    median_int,
    search_band,
    select_guard,
)
from llb.bench.summary_trim.guard_regime import (
    at_guard,
    probe_arm,
    probe_guard,
    scan_guard_band,
    usable_guards,
)
from llb.bench.summary_trim.workloads import build_workload_tasks

GUARD_FIT_FIELD = "middle_critical_guard_fit"
WALK_CONTROL_ARM = "walk_control"

# What the fit concluded, in the same shape the repeated-fold ladder reports its own.
FIT_APPLIED = "guard_fitted_to_the_measured_walk_length"
FIT_DECLARED = "declared_guard_already_folds_within_the_measured_walk"
FIT_UNDERPOWERED = "no_guard_in_the_declared_band_folds_within_every_measured_walk"
FIT_UNMEASURED = "the_walk_control_measured_no_walk_to_fit_against"


def guard_fit_spec(design: dict[str, object]) -> dict[str, object]:
    """The predeclared fit: which workload, which band of guards, and the walk control."""
    return as_mapping(design, GUARD_FIT_FIELD)


def fitted_workload_name(design: dict[str, object]) -> str:
    """The one workload whose guard the fit may move."""
    return as_str(guard_fit_spec(design), "workload")


def walk_control_workload(
    workload: dict[str, object], spec: dict[str, object]
) -> dict[str, object]:
    """The workload as the walk control runs it: declared in every field but the guard."""
    control = as_mapping(spec, "walk_control")
    return {**workload, "max_prompt_chars": as_int(control, "max_prompt_chars")}


def apply_fitted_guard(workload: dict[str, object], record: dict[str, object]) -> dict[str, object]:
    """The workload as it will actually run: declared everywhere except the fitted guard."""
    return at_guard(workload, int(cast(int, record["fitted_max_prompt_chars"])))


def guard_band_reading(design: dict[str, object], workload: dict[str, object]) -> dict[str, object]:
    """The model-free half of the fit: which guards the band can use at all, before any family.

    A family's fit is one score over this set, so the set itself is worth reporting on its own --
    it is what decides the FLOOR every family's fit is bounded by, and it needs no GPU to state.
    """
    spec = guard_fit_spec(design)
    held = as_mapping(design, "held_fixed")
    scan = scan_guard_band(workload, held, spec)
    usable = usable_guards(scan)
    return {
        "workload": workload["workload"],
        "declared_max_prompt_chars": as_int(workload, "max_prompt_chars"),
        "search_band": dict(zip(("min_chars", "max_chars", "step_chars"), search_band(spec))),
        "n_candidates": len(scan),
        "usable_guards": usable,
        "refused_guards": {
            int(cast(int, row["max_prompt_chars"])): row["refusal"]
            for row in scan
            if row["refusal"] is not None
        },
        **_band_floor(scan, usable),
    }


def fit_middle_guard(
    design: dict[str, object],
    workload: dict[str, object],
    held: dict[str, object],
    walk_lengths: dict[str, int],
) -> dict[str, object]:
    """Choose this family's guard for the middle-critical workload, or say why one cannot help."""
    spec = guard_fit_spec(design)
    declared = as_int(workload, "max_prompt_chars")
    scan = scan_guard_band(workload, held, spec)
    usable = usable_guards(scan)
    walks = sorted(walk_lengths.values())
    floor = len(build_workload_tasks(workload))
    record: dict[str, object] = {
        "workload": workload["workload"],
        "declared_max_prompt_chars": declared,
        "fitted_max_prompt_chars": declared,
        "walk_lengths": dict(sorted(walk_lengths.items())),
        "median_walk_length": median_int(walks),
        "evidence_floor": floor,
        "n_usable_guards": len(usable),
        **_band_floor(scan, usable),
    }
    if not walks or not usable:
        return {**record, **_unmeasured(usable, declared, floor)}
    scores = {guard: sum(walk >= step for walk in walks) for guard, step in usable.items()}
    fitted = select_guard(scores, declared)
    best = scores[fitted]
    short = sorted(item for item, walk in walk_lengths.items() if walk < usable[fitted])
    return {
        **record,
        "fitted_max_prompt_chars": fitted,
        "fitted_fold_step": usable[fitted],
        "declared_fold_step": usable.get(declared, 0),
        "predicted_folding_cases": best,
        "declared_folding_cases": scores.get(declared, 0),
        "short_walk_cases": short,
        "meets_evidence_floor": best >= floor,
        "fit_reading": _fit_reading(fitted, declared, best, floor),
        "fit_reason": _fit_reason(record, fitted, usable[fitted], best, short),
    }


def validate_guard_fit(design: dict[str, object], workloads: list[dict[str, object]]) -> None:
    """Refuse a fit block that could not be applied, or a band the declared guard is outside of."""
    spec = guard_fit_spec(design)
    if not spec:
        raise ValueError(f"the adoption design must declare its {GUARD_FIT_FIELD!r} band")
    name = fitted_workload_name(design)
    workload = next((row for row in workloads if row["workload"] == name), None)
    if workload is None:
        raise ValueError(f"the guard fit names {name!r}, which is not a declared workload")
    held = as_mapping(design, "held_fixed")
    declared = as_int(workload, "max_prompt_chars")
    grid = guard_grid(spec)
    if declared not in grid:
        raise ValueError(f"the declared guard {declared} is not a candidate in its own band")
    refusal = probe_guard(workload, held, declared)["refusal"]
    if refusal is not None:
        raise ValueError(f"the declared guard {declared} does not hold the regime: {refusal}")
    _validate_walk_control(workload, held, spec)


def _validate_walk_control(
    workload: dict[str, object], held: dict[str, object], spec: dict[str, object]
) -> None:
    """The control that measures a walk must not fold: a fold would change what it measures."""
    control = walk_control_workload(workload, spec)
    probe = probe_arm(control, held, SUMMARY_TRIM_HEAD_TAIL)
    folds = max(int(cast(int, row["n_compactions"])) for row in probe)
    if folds:
        raise ValueError(
            f"the walk control folds {folds} time(s) at guard {control['max_prompt_chars']}, so it "
            "measures a folded walk rather than the walk that decides whether a fold is reached"
        )


def _band_floor(scan: list[dict[str, object]], usable: dict[int, int]) -> dict[str, object]:
    """The earliest-folding usable guard, and what refused the candidate just below it."""
    if not usable:
        return {"band_floor_guard": 0, "band_floor_fold_step": 0, "band_floor_reason": None}
    floor_guard = min(usable, key=lambda guard: (usable[guard], guard))
    below = [row for row in scan if int(cast(int, row["max_prompt_chars"])) < floor_guard]
    return {
        "band_floor_guard": floor_guard,
        "band_floor_fold_step": usable[floor_guard],
        "band_floor_reason": below[-1]["refusal"] if below else None,
    }


def _unmeasured(usable: dict[int, int], declared: int, floor: int) -> dict[str, object]:
    reason = (
        "the walk control measured no walk"
        if usable
        else "no guard in the declared band holds the workload's regime"
    )
    return {
        "fitted_fold_step": usable.get(declared, 0),
        "declared_fold_step": usable.get(declared, 0),
        "predicted_folding_cases": 0,
        "declared_folding_cases": 0,
        "short_walk_cases": [],
        "meets_evidence_floor": False,
        "fit_reading": FIT_UNMEASURED,
        "fit_reason": f"{reason}, so the declared guard {declared} stands unfitted (floor {floor})",
    }


def _fit_reading(fitted: int, declared: int, best: int, floor: int) -> str:
    if best < floor:
        return FIT_UNDERPOWERED
    return FIT_DECLARED if fitted == declared else FIT_APPLIED


def _fit_reason(
    record: dict[str, object], fitted: int, step: int, best: int, short: list[str]
) -> str:
    """What the fit chose, and -- when it fell short -- which guard came closest and what stopped it.

    A shortfall is only readable if the band's own floor is named beside it: the fit exhausted the
    declared candidates, and the reason the next one down is not a candidate is a property of the
    workload's geometry rather than of the family.
    """
    floor = int(cast(int, record["evidence_floor"]))
    reason = (
        f"guard {fitted} folds at step {step}, which {best} of {floor} declared case(s) walk far "
        "enough to reach"
    )
    if not short:
        return reason
    return (
        f"{reason}; short-walk case(s) {short}. The earliest-folding guard the band can still use "
        f"is {record['band_floor_guard']} at step {record['band_floor_fold_step']}, and the "
        f"candidate below it is refused because {record['band_floor_reason']}"
    )
