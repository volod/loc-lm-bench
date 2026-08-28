"""Two-family execution and conditional entry-aware prototype for window elision."""

from dataclasses import dataclass
from typing import cast

from llb.bench.agentic.context_policy import SUMMARY_TRIM_PER_ENTRY_HEAD
from llb.bench.context_policy.run import task_set_digest
from llb.bench.context_policy.report import PolicyReport
from llb.bench.memory.window_elision.run import (
    WindowElisionRun,
    run_window_elision_cell,
    run_window_elision_tasks,
)
from llb.bench.memory.window_elision.design import ROLE_ELIDED
from llb.bench.memory.window_elision.transfer_design import (
    probe_transfer_cell,
    transfer_case_metadata,
    transfer_cells,
    transfer_placements,
    transfer_tasks,
)
from llb.bench.memory.window_elision.transfer_reading import (
    PROTOTYPE_NOT_RUN,
    family_stratum_reading,
    prototype_reading,
    transfer_reading,
)
from llb.bench.common import LLMComplete


@dataclass(slots=True)
class TransferFamilyRun:
    """One candidate's base comparison plus an optional entry-aware treatment cell."""

    model_family: str
    model: str
    backend: str
    base: WindowElisionRun
    analysis: dict[str, object]
    prototype_report: PolicyReport | None = None
    prototype_row: dict[str, object] | None = None
    tokens_per_s: float = 0.0
    prototype_tokens_per_s: float = 0.0


def run_transfer_family(
    design: dict[str, object],
    candidate: dict[str, object],
    *,
    complete: LLMComplete,
) -> TransferFamilyRun:
    """Run one family control-first through all three evidence strata."""
    tasks = transfer_tasks(design)
    probes = {
        cast(str, cell["cell_id"]): probe_transfer_cell(design, cell)
        for cell in transfer_cells(design)
    }
    model = cast(str, candidate["model"])
    backend = cast(str, candidate["backend"])
    base = run_window_elision_tasks(
        design,
        tasks=tasks,
        model=model,
        backend=backend,
        complete=complete,
        case_metadata=transfer_case_metadata(design),
        cell_probes=probes,
    )
    analysis = family_stratum_reading(base.analysis)
    analysis["model_family"] = candidate["model_family"]
    return TransferFamilyRun(
        model_family=cast(str, candidate["model_family"]),
        model=model,
        backend=backend,
        base=base,
        analysis=analysis,
    )


def run_entry_aware_prototype(
    design: dict[str, object], run: TransferFamilyRun, *, complete: LLMComplete
) -> None:
    """Attach one same-guard entry-aware treatment after the cross-family gate opens."""
    tasks = transfer_tasks(design)
    held = cast(dict[str, object], design["held_fixed"])
    cell = next(cell for cell in transfer_cells(design) if cell["role"] == ROLE_ELIDED)
    report, row = run_window_elision_cell(
        tasks,
        cell,
        held,
        model=run.model,
        backend=run.backend,
        complete=complete,
        probe=probe_transfer_cell(design, cell),
        task_digest=task_set_digest(tasks),
        case_metadata=transfer_case_metadata(design),
        summary_trim_strategy=SUMMARY_TRIM_PER_ENTRY_HEAD,
    )
    row["summary_trim_strategy"] = SUMMARY_TRIM_PER_ENTRY_HEAD
    run.prototype_report = report
    run.prototype_row = row


def analyze_transfer_runs(
    design: dict[str, object], runs: list[TransferFamilyRun]
) -> dict[str, object]:
    """Read cross-family strata and, when present, the gated prototype."""
    required = int(cast(int, design["required_qualified_families"]))
    families = [run.analysis for run in runs]
    reading, reason, prototype_required, qualified = transfer_reading(
        families, required_families=required
    )
    prototype_rows = _prototype_rows(runs, qualified)
    if prototype_required and len(prototype_rows) == len(qualified):
        prototype, prototype_reason, prototype_detail = prototype_reading(qualified, prototype_rows)
    else:
        prototype = PROTOTYPE_NOT_RUN
        prototype_reason = (
            "the middle-specific cross-family gate did not open"
            if not prototype_required
            else "the required entry-aware prototype cells have not all run"
        )
        prototype_detail = {"families": []}
    return {
        "study_id": design["study_id"],
        "study_kind": design["study_kind"],
        "seed": design["seed"],
        "required_qualified_families": required,
        "families": families,
        "qualified_models": [row["model"] for row in qualified],
        "placements": transfer_placements(design),
        "transfer_reading": reading,
        "transfer_reason": reason,
        "prototype_required": prototype_required,
        "prototype_reading": prototype,
        "prototype_reason": prototype_reason,
        "prototype_detail": prototype_detail,
        "changes_shipped_default": False,
    }


def _prototype_rows(
    runs: list[TransferFamilyRun], qualified: list[dict[str, object]]
) -> list[dict[str, object]]:
    qualified_models = {cast(str, row["model"]) for row in qualified}
    rows: list[dict[str, object]] = []
    for run in runs:
        if run.model not in qualified_models or run.prototype_row is None:
            continue
        head_tail = next(
            row
            for row in cast(list[dict[str, object]], run.base.analysis["cells"])
            if row["role"] == ROLE_ELIDED
        )
        rows.append(
            {
                "model": run.model,
                "head_tail_cases": head_tail["cases"],
                "entry_aware_cases": run.prototype_row["cases"],
            }
        )
    return rows
