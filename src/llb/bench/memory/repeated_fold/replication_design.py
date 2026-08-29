"""Design contract for the two-family repeated-fold completion replication.

The completion reading it replicates was taken on one model family over two memory cases
(`agentic_compact_repeated_fold_completion_design.json`). This design holds the three fold-count
cells and the marker ablation EXACTLY as that one declares them -- what it adds is a larger
predeclared case set, a candidate roster the eligibility gate qualifies families from, and a
per-fold paired-evidence floor. Nothing here may move the geometry: the shared cell contract is
validated by the completion design's own validator so a drift shows up as one failure, not two.
"""

import hashlib
from pathlib import Path
from typing import cast

from llb.bench.agentic.design_fields import as_int, as_mapping, as_rows, as_str
from llb.bench.memory.repeated_fold.design import (
    completion_cells,
    probe_completion_cell,
    validate_repeated_fold_design,
)
from llb.bench.context_policy.guard_band import search_band
from llb.bench.memory.repeated_fold.guard_fit import GUARD_FIT_FIELD, guard_fit_spec
from llb.bench.policy_change.geometry import load_audited_design

DESIGN_PATH = "samples/benchmarks/agentic_compact_repeated_fold_replication_design.json"
STUDY_KIND = "compact_repeated_fold_replication"
# Two families is the whole point of a replication: one family cannot separate a fold-count rule
# from a property of the model that produced it.
REQUIRED_FAMILIES = 2
# The completion design this replicates ran two cases per cell; a replication that did not raise
# that number would restate the same ceiling with the same power.
MIN_REPLICATION_TASKS = 8


def load_repeated_fold_replication_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed replication design through the shared strict JSON loader."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / DESIGN_PATH if path is None else path)


def replication_roster(design: dict[str, object]) -> list[dict[str, object]]:
    """Candidate families in the order the eligibility gate should try them."""
    return as_rows(design, "candidate_roster")


def roster_digest(roster: list[dict[str, object]]) -> str:
    """Content digest of the families a run actually drove, in roster order."""
    payload = "|".join(f"{row['model_family']}:{row['model']}:{row['backend']}" for row in roster)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def minimum_paired_cases(design: dict[str, object]) -> int:
    """The predeclared paired-evidence floor a measured fold group must reach to be read."""
    held = as_mapping(design, "held_fixed")
    return int(cast(int, held["minimum_paired_cases_per_fold"]))


def validate_replication_design(design: dict[str, object]) -> None:
    """Require the cell contract, a larger case set, a roster, a floor, and a per-family fit."""
    validate_repeated_fold_design(design, study_kind=STUDY_KIND)
    _validate_case_set(design)
    _validate_roster(design)
    if not guard_fit_spec(design):
        raise ValueError(
            f"a repeated-fold replication must declare a {GUARD_FIT_FIELD!r} block: one shared "
            "char guard puts a family whose summaries run long on the wrong rung of the ladder"
        )
    _validate_guard_fit(design, completion_cells(design))


def _validate_case_set(design: dict[str, object]) -> None:
    held = as_mapping(design, "held_fixed")
    n_tasks = int(cast(int, held.get("n_tasks", 0)))
    if n_tasks < MIN_REPLICATION_TASKS:
        raise ValueError(
            f"a repeated-fold replication needs at least {MIN_REPLICATION_TASKS} predeclared "
            f"cases, got {n_tasks}"
        )
    floor = int(cast(int, held.get("minimum_paired_cases_per_fold", 0)))
    if not 1 < floor <= n_tasks:
        raise ValueError("the paired-evidence floor must be above one and within the case set")


def _validate_roster(design: dict[str, object]) -> None:
    required = int(cast(int, design.get("required_qualified_families", 0)))
    roster = replication_roster(design)
    families = [str(row.get("model_family", "")) for row in roster]
    models = [str(row.get("model", "")) for row in roster]
    if required != REQUIRED_FAMILIES or len(roster) < required:
        raise ValueError(
            f"the repeated-fold replication requires {REQUIRED_FAMILIES} qualified model families"
        )
    if not all(families) or len(families) != len(set(families)):
        raise ValueError("every replication candidate must name a distinct model family")
    if not all(models) or len(models) != len(set(models)):
        raise ValueError("every replication candidate must name a distinct model")
    if any(row.get("backend") != "ollama" for row in roster):
        raise ValueError("the repeated-fold replication roster must use local Ollama models")


def _validate_guard_fit(design: dict[str, object], cells: list[dict[str, object]]) -> None:
    """Refuse a fit that could move the cell out of the regime the ladder reads it in.

    The band is predeclared for the same reason the cells are: a guard search free to run to the
    cap peak could hand a family a CAP-FITTING guard, where the fold-count rung is unreachable by
    construction, and a search free to run below the deeper cell's guard could hand it that cell's
    regime instead. Both bounds are checked here rather than discovered in the run.
    """
    spec = guard_fit_spec(design)
    if not spec:
        return
    cell = _fitted_cell(spec, cells)
    target = as_int(spec, "target_folds")
    if bool(cell.get("cap_fitting_control")):
        raise ValueError("the one-fold cap-fitting control anchors the ladder and is never fitted")
    if target != as_int(cell, "expected_oracle_folds"):
        raise ValueError(
            f"the fitted cell {cell['cell_id']!r} declares "
            f"{cell['expected_oracle_folds']} oracle folds, not the fitted target {target}"
        )
    if target < 2:
        raise ValueError("a one-fold rung is the control's job; fit a repeatedly folding cell")
    _validate_band(spec, cell, cells, as_mapping(design, "held_fixed"))
    source = as_str(spec, "fold_length_source")
    if source not in {str(row["cell_id"]) for row in cells if row.get("cap_fitting_control")}:
        raise ValueError(
            f"the fold length must be measured on a cap-fitting control cell, got {source!r}"
        )


def _validate_band(
    spec: dict[str, object],
    cell: dict[str, object],
    cells: list[dict[str, object]],
    held: dict[str, object],
) -> None:
    low, high, step = search_band(spec)
    if step < 1 or low >= high:
        raise ValueError("the guard search band must be a non-empty ascending range with a step")
    declared = as_int(cell, "max_prompt_chars")
    if not low <= declared <= high:
        raise ValueError(
            f"the guard search band [{low}, {high}] must contain the declared guard {declared}, "
            "or the fit cannot reproduce the shared geometry when a family's summaries are short"
        )
    deeper = [
        as_int(row, "max_prompt_chars")
        for row in cells
        if as_int(row, "expected_oracle_folds") > as_int(cell, "expected_oracle_folds")
    ]
    if deeper and low <= max(deeper):
        raise ValueError(
            f"the guard search band starts at {low}, at or below the deeper cell's guard "
            f"{max(deeper)}, so the fit could hand one cell the other's regime"
        )
    peak = int(cast(int, probe_completion_cell(cell, held)["cap_peak_prompt_chars"]))
    if high >= peak:
        raise ValueError(
            f"the guard search band ends at {high}, at or above the {peak}-char cap peak, where "
            "the cell is cap-fitting and folds once by construction"
        )


def _fitted_cell(spec: dict[str, object], cells: list[dict[str, object]]) -> dict[str, object]:
    cell_id = as_str(spec, "cell_id")
    matched = [row for row in cells if row.get("cell_id") == cell_id]
    if not matched:
        raise ValueError(f"the guard fit names cell {cell_id!r}, which the design does not declare")
    return matched[0]
