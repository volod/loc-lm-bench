"""Fitting one fold-count cell's prompt guard to the family that will run it.

A fold-count ladder holds the GUARD fixed and reads whatever fold count each family lands on. That
works while every family writes summaries of a similar length and stops working the moment one does
not: the running summary sits in every later prompt, so a family whose summaries run long spends
more of the guard before the transcript grows at all, re-crosses the compact trigger sooner, and
folds again on the geometry another family folds twice on. The rung is then empty for that family
-- not because the fold count is hard to reach, but because one shared character constant was fit
against a summarizer nobody ran.

What this module fits instead is the guard, per family, from that family's OWN measured fold length.
The measurement is free: the one-fold control already folds exactly once per case, so its telemetry
carries one summarizer output length per case before any other cell runs. The fit itself is
model-free -- the deterministic oracle walk, replayed with a summarizer that writes exactly the
measured number of characters (`fold_length_controller`), over a predeclared band of candidate
guards. What is held equal across families is therefore the RUNG -- the measured fold count -- and
the guard is what each family needs to stand on it.
"""

from typing import Callable, cast

from llb.bench.agentic.design_fields import as_int, as_mapping, as_str
from llb.bench.memory.boundary.probe import compact_fold_input_probe, fold_length_controller
from llb.bench.memory.repeated_fold.design import cell_geometry

GUARD_FIT_FIELD = "two_fold_guard_fit"
ARM_TYPED_MARKER = "typed_marker"

FIT_APPLIED = "guard_fitted_to_the_measured_fold_length"
FIT_DECLARED = "declared_guard_already_fits_the_measured_fold_length"
FIT_UNDERPOWERED = "no_guard_in_the_declared_band_reaches_the_evidence_floor"
FIT_UNMEASURED = "the_control_measured_no_fold_length_to_fit_against"


def guard_fit_spec(design: dict[str, object]) -> dict[str, object]:
    """The predeclared fit: which cell, which fold count, and which band of guards to search."""
    return as_mapping(design, GUARD_FIT_FIELD)


def search_band(spec: dict[str, object]) -> tuple[int, int, int]:
    """The predeclared candidate guards, as an inclusive ascending range with a step."""
    return (
        as_int(spec, "search_min_chars"),
        as_int(spec, "search_max_chars"),
        as_int(spec, "step_chars"),
    )


def measured_fold_lengths(rows: list[dict[str, object]], source_cell_id: str) -> list[int]:
    """Every summarizer output length the shipped-policy control arm actually measured.

    One length per FOLD, not per case: the control folds once per case, so the two coincide there,
    and a source cell that folded more would contribute each fold it made.
    """
    return [
        int(chars)
        for row in rows
        if row["cell_id"] == source_cell_id and row["arm"] == ARM_TYPED_MARKER
        for case in cast(list[dict[str, object]], row["cases"])
        for chars in cast(list[int], case.get("summary_output_chars", []))
    ]


def fit_fold_guard(
    design: dict[str, object],
    cell: dict[str, object],
    held: dict[str, object],
    fold_lengths: list[int],
    *,
    evidence_floor: int,
) -> dict[str, object]:
    """Choose this family's guard for one fold-count cell, or say why the declared one stands."""
    spec = guard_fit_spec(design)
    declared = as_int(cell, "max_prompt_chars")
    target = as_int(spec, "target_folds")
    record: dict[str, object] = {
        "cell_id": cell["cell_id"],
        "target_folds": target,
        "declared_max_prompt_chars": declared,
        "fitted_max_prompt_chars": declared,
        "fold_length_source": as_str(spec, "fold_length_source"),
        "fold_lengths": fold_lengths,
        "median_fold_length_chars": _median(fold_lengths),
        "evidence_floor": evidence_floor,
    }
    if not fold_lengths:
        return {
            **record,
            "predicted_target_cases": 0,
            "declared_target_cases": 0,
            "meets_evidence_floor": False,
            "fit_reading": FIT_UNMEASURED,
            "fit_reason": (
                f"cell {record['fold_length_source']!r} measured no summarizer output, so the "
                f"declared guard {declared} stands unfitted"
            ),
        }
    predict = _fold_count_predictor(cell, held)
    counts = {guard: _target_cases(predict, guard, fold_lengths, target) for guard in _grid(spec)}
    fitted = _best_guard(counts, declared)
    best = counts[fitted]
    return {
        **record,
        "fitted_max_prompt_chars": fitted,
        "predicted_target_cases": best,
        "declared_target_cases": counts.get(declared, 0),
        "meets_evidence_floor": best >= evidence_floor,
        "fit_reading": _fit_reading(fitted, declared, best, evidence_floor),
        "fit_reason": (
            f"guard {fitted} puts {best} of {len(fold_lengths)} measured fold lengths on "
            f"{target} folds (declared guard {declared} puts {counts.get(declared, 0)}); "
            f"floor {evidence_floor}"
        ),
    }


def apply_fitted_guard(cell: dict[str, object], record: dict[str, object]) -> dict[str, object]:
    """The cell as it will actually run: declared in every field except the fitted guard."""
    return {**cell, "max_prompt_chars": int(cast(int, record["fitted_max_prompt_chars"]))}


def guard_resolver(
    design: dict[str, object], *, evidence_floor: int
) -> (
    Callable[
        [dict[str, object], list[dict[str, object]]],
        tuple[dict[str, object], dict[str, object] | None],
    ]
    | None
):
    """The completion runner's guard seam, or `None` when the design declares no fit.

    Every cell but the fitted one is handed back exactly as declared, so a design that adds a fit
    changes the geometry of ONE rung and leaves the control and the deeper cell untouched.
    """
    spec = guard_fit_spec(design)
    if not spec:
        return None
    held = as_mapping(design, "held_fixed")
    fitted_cell_id = as_str(spec, "cell_id")
    source = as_str(spec, "fold_length_source")

    def resolve(
        cell: dict[str, object], rows: list[dict[str, object]]
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        if cell.get("cell_id") != fitted_cell_id:
            return cell, None
        lengths = measured_fold_lengths(rows, source)
        record = fit_fold_guard(design, cell, held, lengths, evidence_floor=evidence_floor)
        return apply_fitted_guard(cell, record), record

    return resolve


def _fit_reading(fitted: int, declared: int, best: int, floor: int) -> str:
    if best < floor:
        return FIT_UNDERPOWERED
    return FIT_DECLARED if fitted == declared else FIT_APPLIED


def _grid(spec: dict[str, object]) -> list[int]:
    low, high, step = search_band(spec)
    return list(range(low, high + 1, step))


def _fold_count_predictor(
    cell: dict[str, object], held: dict[str, object]
) -> Callable[[int, int], int]:
    """Fold count as a pure function of (guard, fold length), memoized over the search.

    The walk is deterministic and every task in the set produces the same fold count for a given
    pair, so one probe answers for the whole case set and the grid costs one probe per distinct
    pair rather than one per case.
    """
    cache: dict[tuple[int, int], int] = {}

    def predict(guard: int, fold_length: int) -> int:
        key = (guard, fold_length)
        if key not in cache:
            geometry = cell_geometry({**cell, "max_prompt_chars": guard}, held)
            probe = compact_fold_input_probe(
                controller=fold_length_controller(fold_length),
                **geometry,  # type: ignore[arg-type]
            )
            cache[key] = int(cast(int, probe["n_compactions"]))
        return cache[key]

    return predict


def _target_cases(
    predict: Callable[[int, int], int], guard: int, fold_lengths: list[int], target: int
) -> int:
    return sum(predict(guard, length) == target for length in fold_lengths)


def _best_guard(counts: dict[int, int], declared: int) -> int:
    """The centre of the WIDEST run of guards that lands the most cases on the target rung.

    Two tie-breaks, in order. The DECLARED guard wins any tie, so a family the shared constant
    already suits keeps the published geometry and only the family it does not suit moves -- a fit
    that shuffled every guard would invalidate the comparison it exists to enable. Among the rest,
    the middle of the widest contiguous run wins: that is the guard furthest from the length at
    which one more case folds again, so a family whose next run writes slightly different summaries
    still lands on the same rung.
    """
    best = max(counts.values())
    if counts.get(declared) == best:
        return declared
    guards = sorted(counts)
    runs: list[list[int]] = []
    for index, guard in enumerate(guards):
        if counts[guard] != best:
            continue
        contiguous = index > 0 and counts[guards[index - 1]] == best
        if contiguous and runs:
            runs[-1].append(guard)
        else:
            runs.append([guard])
    widest = max(runs, key=len)
    return widest[len(widest) // 2]


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
