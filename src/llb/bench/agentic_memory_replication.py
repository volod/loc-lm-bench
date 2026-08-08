"""Second-family, tighter-evidence replication of compact-memory transfer."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_boundary_gate import (
    boundary_cost_evidence,
    clears_tighter_completion,
)
from llb.bench.agentic_memory_transfer import load_transfer_design
from llb.bench.agentic_memory_transfer_cells import (
    held_summary_input_cap,
    run_transfer_cell,
)
from llb.bench.agentic_design_fields import (
    as_bool,
    as_float,
    as_int,
    as_mapping,
    as_rows,
    as_str,
    as_strs,
)
from llb.bench.common import LLMComplete
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs

STUDY_KIND = "compact_memory_transfer_replication"
METHOD = "agentic-compact-memory-transfer-replication"

READING_INELIGIBLE = "second_family_control_ineligible"
READING_NOT_REPRODUCED = "second_family_transfer_not_reproduced"
READING_BOUNDARY_FAILED = "cap_fitting_boundary_invalid"
READING_BOUNDARY_COMPACT = "replicated_including_cap_fitting"
READING_BOUNDARY_CAP = "replicated_overflow_gain_cap_fitting_prefers_cap"
READING_BOUNDARY_TIED = "replicated_overflow_gain_cap_fitting_tied"


def load_replication_design(path: Path | str) -> dict[str, object]:
    """Reuse the strict JSON loader while applying replication-specific validation later."""
    return load_transfer_design(path)


def _check_replication_identity(design: dict[str, object]) -> None:
    """The design is the replication schema this validator was written against."""
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("compact-memory replication design schema or study kind is invalid")


def _check_second_family_roster(design: dict[str, object]) -> None:
    """The candidates are new, unique, non-Qwen families -- which is what makes it a replication."""
    references = set(as_strs(design, "reference_model_families"))
    candidates = as_rows(design, "candidate_roster")
    families = [as_str(row, "model_family") for row in candidates]
    models = [as_str(row, "model") for row in candidates]
    if not references or not candidates or len(families) != len(set(families)):
        raise ValueError("replication needs reference families and unique candidate families")
    if any(
        not family
        or family in references
        or "qwen" in family.lower()
        or not model
        or "qwen" in model.lower()
        for family, model in zip(families, models, strict=True)
    ):
        raise ValueError("replication candidates must be new non-Qwen model families")
    if any(row.get("backend") != "ollama" for row in candidates):
        raise ValueError("the local replication roster must use Ollama")


def _check_control_pilot(design: dict[str, object]) -> None:
    """The unchanged token-chain control that decides which candidate family is even eligible."""
    control = as_mapping(design, "control_pilot")
    threshold = as_float(control, "minimum_completion_rate")
    if as_int(control, "n_tasks") < 1 or as_int(control, "depth") < 3 or not 0.0 < threshold <= 1.0:
        raise ValueError("replication control pilot contract is invalid")


def _check_tighter_reading_power(design: dict[str, object]) -> None:
    """The tighter reading this study exists for: 97.5%, and enough tasks to reach it."""
    if as_float(design, "reporting_confidence") != 0.975:
        raise ValueError("replication reporting confidence must be exactly 0.975")
    matrix = as_mapping(design, "matrix")
    if as_int(matrix, "n_tasks") < minimum_discordant_pairs(0.975):
        raise ValueError("replication cells cannot reach the 97.5% exact-sign-test floor")


def _check_cell_list(design: dict[str, object]) -> None:
    """Four transfer cells and one cap-fitting boundary, each uniquely named and runnable."""
    cells = as_rows(as_mapping(design, "matrix"), "cells")
    ids = [as_str(cell, "cell_id") for cell in cells]
    if len(cells) < 5 or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("replication needs at least five uniquely named cells")
    boundaries = [cell for cell in cells if as_bool(cell, "require_cap_fits")]
    if len(boundaries) != 1 or len(cells) - len(boundaries) < 4:
        raise ValueError("replication needs four transfer cells and one cap-fitting boundary")
    if any(
        as_int(cell, "depth") < 3
        or not 0.0 < as_float(cell, "compact_share") <= 1.0
        or as_int(cell, "max_prompt_chars") <= 0
        for cell in cells
    ):
        raise ValueError("replication cell geometry is invalid")


def validate_replication_design(design: dict[str, object]) -> None:
    """Require a second family, 97.5%-capable cells, and a cap-fitting boundary."""
    _check_replication_identity(design)
    _check_second_family_roster(design)
    _check_control_pilot(design)
    _check_tighter_reading_power(design)
    _check_cell_list(design)


def run_replication_matrix(
    design: dict[str, object], *, model: str, backend: str, complete: LLMComplete, data_dir: Path
) -> list[dict[str, object]]:
    """Run the exact heterogeneous replication cell list in declared order."""
    matrix = cast(dict[str, object], design["matrix"])
    return [
        run_transfer_cell(
            model=model,
            backend=backend,
            complete=complete,
            data_dir=data_dir,
            n_tasks=int(cast(int, matrix["n_tasks"])),
            pad_chars=int(cast(int, matrix["pad_chars"])),
            max_steps_margin=int(cast(int, matrix["max_steps_margin"])),
            observation_cap_chars=int(cast(int, matrix["observation_cap_chars"])),
            observation_head_share=float(cast(float, matrix["observation_head_share"])),
            minimum_compaction_rate=float(cast(float, matrix["minimum_compaction_rate"])),
            summary_input_cap=held_summary_input_cap(matrix),
            cell_id=cast(str, cell["cell_id"]),
            depth=int(cast(int, cell["depth"])),
            compact_share=float(cast(float, cell["compact_share"])),
            max_prompt_chars=int(cast(int, cell["max_prompt_chars"])),
            require_cap_fits=bool(cell["require_cap_fits"]),
        )
        for cell in cast(list[dict[str, object]], matrix["cells"])
    ]


@dataclass(frozen=True, slots=True)
class _ReplicationVerdict:
    """What the replication cells said, and the reading that follows from it."""

    core_passed: bool
    boundary_passed: bool
    boundary_evidence: dict[str, object] | None
    reading: str
    reason: str


def _ineligible_verdict() -> _ReplicationVerdict:
    """No candidate family passed the control, so nothing downstream may be read at all."""
    return _ReplicationVerdict(
        core_passed=False,
        boundary_passed=False,
        boundary_evidence=None,
        reading=READING_INELIGIBLE,
        reason="no new non-Qwen family passed the unchanged token-chain control",
    )


def _boundary_reading(evidence: dict[str, object]) -> tuple[str, str]:
    """Which way the cap-fitting boundary points once both arms fit."""
    if evidence["compact_preference_clears"]:
        return (
            READING_BOUNDARY_COMPACT,
            "compact replicates at 97.5% and remains preferred when cap fits",
        )
    if evidence["cap_preference_clears"]:
        return (
            READING_BOUNDARY_CAP,
            "compact replicates under overflow, but cap is cheaper when both fit",
        )
    return (
        READING_BOUNDARY_TIED,
        "compact replicates under overflow; the cap-fitting boundary is tied",
    )


def _measured_verdict(
    design: dict[str, object], matrix_rows: list[dict[str, object]]
) -> _ReplicationVerdict:
    """Apply the tighter completion gate to the transfer cells, then the cap-fitting boundary."""
    confidence = as_float(design, "reporting_confidence")
    core = [row for row in matrix_rows if not row["require_cap_fits"]]
    boundary = next(row for row in matrix_rows if row["require_cap_fits"])
    core_passed = all(clears_tighter_completion(row, confidence) for row in core)
    minimum_rate = as_float(as_mapping(design, "matrix"), "minimum_compaction_rate")
    boundary_passed = bool(
        boundary["cap_context_overflows"] == 0
        and cast(float, boundary["compaction_activation_rate"]) >= minimum_rate
    )
    evidence = boundary_cost_evidence(boundary, confidence)
    if not core_passed:
        reading, reason = (
            READING_NOT_REPRODUCED,
            "the second family does not reproduce every transfer cell at 97.5%",
        )
    elif not boundary_passed:
        reading, reason = (
            READING_BOUNDARY_FAILED,
            "the boundary did not keep cap usable while activating compact",
        )
    else:
        reading, reason = _boundary_reading(evidence)
    return _ReplicationVerdict(
        core_passed=core_passed,
        boundary_passed=boundary_passed,
        boundary_evidence=evidence,
        reading=reading,
        reason=reason,
    )


def analyze_replication(
    design: dict[str, object],
    pilot_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the second-family, tighter-reading, and cap-fitting gates."""
    selected = next((row for row in pilot_rows if row["eligible"]), None)
    if selected is None:
        verdict = _ineligible_verdict()
    else:
        expected = as_rows(as_mapping(design, "matrix"), "cells")
        if [row["cell_id"] for row in matrix_rows] != [cell["cell_id"] for cell in expected]:
            raise ValueError("replication rows do not match the exact declared cell order")
        verdict = _measured_verdict(design, matrix_rows)
    return {
        "study_id": design["study_id"],
        "selected_candidate": selected,
        "control_pilots": pilot_rows,
        "matrix_rows": matrix_rows,
        "tighter_transfer_cells_passed": verdict.core_passed,
        "cap_fitting_boundary_passed": verdict.boundary_passed,
        "boundary_directional_evidence": verdict.boundary_evidence,
        "replication_reading": verdict.reading,
        "reason": verdict.reason,
        "changes_shipped_default": False,
    }
