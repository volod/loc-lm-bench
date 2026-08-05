"""Second-family, tighter-evidence replication of compact-memory transfer."""

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


def validate_replication_design(design: dict[str, object]) -> None:
    """Require a second family, 97.5%-capable cells, and a cap-fitting boundary."""
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("compact-memory replication design schema or study kind is invalid")
    references = set(cast(list[str], design.get("reference_model_families", [])))
    candidates = cast(list[dict[str, object]], design.get("candidate_roster", []))
    families = [str(row.get("model_family", "")) for row in candidates]
    models = [str(row.get("model", "")) for row in candidates]
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

    control = cast(dict[str, object], design.get("control_pilot", {}))
    threshold = float(cast(float, control.get("minimum_completion_rate", 0.0)))
    if (
        int(cast(int, control.get("n_tasks", 0))) < 1
        or int(cast(int, control.get("depth", 0))) < 3
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("replication control pilot contract is invalid")

    confidence = float(cast(float, design.get("reporting_confidence", 0.0)))
    if confidence != 0.975:
        raise ValueError("replication reporting confidence must be exactly 0.975")
    matrix = cast(dict[str, object], design.get("matrix", {}))
    n_tasks = int(cast(int, matrix.get("n_tasks", 0)))
    if n_tasks < minimum_discordant_pairs(confidence):
        raise ValueError("replication cells cannot reach the 97.5% exact-sign-test floor")
    cells = cast(list[dict[str, object]], matrix.get("cells", []))
    ids = [str(cell.get("cell_id", "")) for cell in cells]
    if len(cells) < 5 or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("replication needs at least five uniquely named cells")
    boundaries = [cell for cell in cells if cell.get("require_cap_fits") is True]
    references_cells = [cell for cell in cells if cell.get("require_cap_fits") is False]
    if len(boundaries) != 1 or len(references_cells) < 4:
        raise ValueError("replication needs four transfer cells and one cap-fitting boundary")
    if any(
        int(cast(int, cell.get("depth", 0))) < 3
        or not 0.0 < float(cast(float, cell.get("compact_share", 0.0))) <= 1.0
        or int(cast(int, cell.get("max_prompt_chars", 0))) <= 0
        for cell in cells
    ):
        raise ValueError("replication cell geometry is invalid")


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


def analyze_replication(
    design: dict[str, object],
    pilot_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the second-family, tighter-reading, and cap-fitting gates."""
    selected = next((row for row in pilot_rows if row["eligible"]), None)
    expected = cast(list[dict[str, object]], cast(dict[str, object], design["matrix"])["cells"])
    if selected is None:
        reading = READING_INELIGIBLE
        reason = "no new non-Qwen family passed the unchanged token-chain control"
        core_passed = False
        boundary_passed = False
        boundary_evidence = None
    elif [row["cell_id"] for row in matrix_rows] != [cell["cell_id"] for cell in expected]:
        raise ValueError("replication rows do not match the exact declared cell order")
    else:
        confidence = float(cast(float, design["reporting_confidence"]))
        core = [row for row in matrix_rows if not row["require_cap_fits"]]
        boundary = next(row for row in matrix_rows if row["require_cap_fits"])
        core_passed = all(clears_tighter_completion(row, confidence) for row in core)
        minimum_rate = float(
            cast(float, cast(dict[str, object], design["matrix"])["minimum_compaction_rate"])
        )
        boundary_passed = bool(
            boundary["cap_context_overflows"] == 0
            and cast(float, boundary["compaction_activation_rate"]) >= minimum_rate
        )
        boundary_evidence = boundary_cost_evidence(boundary, confidence)
        if not core_passed:
            reading = READING_NOT_REPRODUCED
            reason = "the second family does not reproduce every transfer cell at 97.5%"
        elif not boundary_passed:
            reading = READING_BOUNDARY_FAILED
            reason = "the boundary did not keep cap usable while activating compact"
        elif boundary_evidence["compact_preference_clears"]:
            reading = READING_BOUNDARY_COMPACT
            reason = "compact replicates at 97.5% and remains preferred when cap fits"
        elif boundary_evidence["cap_preference_clears"]:
            reading = READING_BOUNDARY_CAP
            reason = "compact replicates under overflow, but cap is cheaper when both fit"
        else:
            reading = READING_BOUNDARY_TIED
            reason = "compact replicates under overflow; the cap-fitting boundary is tied"
    return {
        "study_id": design["study_id"],
        "selected_candidate": selected,
        "control_pilots": pilot_rows,
        "matrix_rows": matrix_rows,
        "tighter_transfer_cells_passed": core_passed,
        "cap_fitting_boundary_passed": boundary_passed,
        "boundary_directional_evidence": boundary_evidence,
        "replication_reading": reading,
        "reason": reason,
        "changes_shipped_default": False,
    }
