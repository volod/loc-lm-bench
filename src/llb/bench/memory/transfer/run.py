"""Eligibility gate and depth/trigger matrix for compact-memory transfer."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import POLICY_OBSERVATION_CAP, ContextPolicy
from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.compact_vs_cap_report import (
    VERDICT_PREFER_CAP,
    VERDICT_PREFER_COMPACT,
)
from llb.bench.context_policy.run import run_policy, task_set_digest
from llb.bench.agentic.design_fields import (
    as_float,
    as_floats,
    as_int,
    as_ints,
    as_mapping,
    as_rows,
    as_str,
)
from llb.bench.context_policy.report import METRIC_TOTAL_MODEL_INPUT_TOKENS, PolicyReport
from llb.bench.memory.transfer.cells import (
    held_summary_input_cap,
    run_transfer_cell,
)
from llb.bench.memory.transcript import (
    build_token_chain_control_tasks,
)
from llb.bench.common import LLMComplete

DESIGN_SCHEMA_VERSION = 1
STUDY_KIND = "compact_memory_cross_model_transfer"
METHOD = "agentic-compact-memory-transfer"

READING_INELIGIBLE = "control_ineligible"
READING_PORTABLE = "portable_across_matrix"
READING_CONDITIONAL = "geometry_dependent_transfer"
READING_NOT_REPRODUCED = "transfer_not_reproduced"


def load_transfer_design(path: Path | str) -> dict[str, object]:
    """Load one prospective transfer design."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read compact-memory transfer design {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("compact-memory transfer design must be a JSON object")
    return cast(dict[str, object], raw)


def _check_transfer_identity(design: dict[str, object]) -> None:
    """The design is the transfer schema and study this validator was written against."""
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ValueError(f"transfer design schema_version must be {DESIGN_SCHEMA_VERSION}")
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"transfer design study_kind must be {STUDY_KIND}")


def _check_candidate_roster(design: dict[str, object]) -> None:
    """Unique non-Qwen families on the local backend -- the point of a transfer study."""
    candidates = as_rows(design, "candidate_roster")
    identities = [(as_str(row, "model_family"), as_str(row, "model")) for row in candidates]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("candidate roster must contain unique model-family/model pairs")
    if any(
        not family or not model or "qwen" in family.lower() or "qwen" in model.lower()
        for family, model in identities
    ):
        raise ValueError("every transfer candidate must name one non-Qwen family and model")
    if any(row.get("backend") != "ollama" for row in candidates):
        raise ValueError("the local transfer candidate roster must use Ollama")


def _check_control_pilot(design: dict[str, object]) -> None:
    """The memory-free control block that decides eligibility before any matrix cell runs."""
    control = as_mapping(design, "control_pilot")
    if as_int(control, "n_tasks") < 1:
        raise ValueError("control pilot needs at least one task")
    if as_int(control, "depth") < 3:
        raise ValueError("control pilot depth must be at least 3")
    if not 0.0 < as_float(control, "minimum_completion_rate") <= 1.0:
        raise ValueError("control pilot completion threshold must be in (0, 1]")


def _check_matrix_depths(matrix: dict[str, object]) -> None:
    """The depths must bracket the reference depth, or the matrix cannot answer portability."""
    depths = as_ints(matrix, "depths")
    reference_depth = as_int(matrix, "reference_depth")
    if (
        len(depths) < 2
        or len(set(depths)) != len(depths)
        or min(depths) >= reference_depth
        or max(depths) <= reference_depth
    ):
        raise ValueError("matrix depths must uniquely bracket the reference depth")
    if any(depth < 3 for depth in depths):
        raise ValueError("every matrix depth must be at least 3")


def _check_matrix_shares(matrix: dict[str, object]) -> None:
    """The compact shares must bracket the reference trigger, for the same reason."""
    shares = as_floats(matrix, "compact_shares")
    reference_share = as_float(matrix, "reference_compact_share")
    if len(shares) < 2 or len(set(shares)) != len(shares):
        raise ValueError("matrix compact shares must contain at least two unique values")
    brackets = min(shares) < reference_share < max(shares)
    if any(not 0.0 < share <= 1.0 for share in shares) or not brackets:
        raise ValueError("matrix compact shares must bracket the reference trigger")


def _check_matrix_power(matrix: dict[str, object]) -> None:
    """Enough paired tasks per cell to read a 95% sign test, and a legible compaction floor."""
    if as_int(matrix, "n_tasks") < 6:
        raise ValueError("each matrix cell needs at least six paired tasks for 95% evidence")
    if not 0.0 <= as_float(matrix, "minimum_compaction_rate", -1.0) <= 1.0:
        raise ValueError("matrix minimum compaction rate must be in [0, 1]")


def validate_transfer_design(design: dict[str, object]) -> None:
    """Refuse an underpowered, Qwen, or non-bracketing transfer contract."""
    _check_transfer_identity(design)
    _check_candidate_roster(design)
    _check_control_pilot(design)
    matrix = as_mapping(design, "matrix")
    _check_matrix_depths(matrix)
    _check_matrix_shares(matrix)
    _check_matrix_power(matrix)


def run_control_pilot(
    control: dict[str, object], *, model: str, backend: str, complete: LLMComplete
) -> tuple[PolicyReport, dict[str, object]]:
    """Run the memory-free token chain for ONE control block and return its eligibility record."""
    tasks = [
        AgenticTask.from_record(row)
        for row in build_token_chain_control_tasks(
            n_tasks=int(cast(int, control["n_tasks"])),
            depth=int(cast(int, control["depth"])),
        )
    ]
    report = run_policy(
        tasks,
        ContextPolicy(name=POLICY_OBSERVATION_CAP),
        model=model,
        backend=backend,
        complete=complete,
        max_steps=int(cast(int, control["max_steps"])),
        budget=fixed_budget(int(cast(int, control["max_prompt_chars"]))),
    )
    completion = report.result.objective_score
    row: dict[str, object] = {
        "model": model,
        "backend": backend,
        "task_set_digest": task_set_digest(tasks),
        "n_tasks": len(tasks),
        "depth": control["depth"],
        "completion": completion,
        "minimum_completion_rate": control["minimum_completion_rate"],
        "eligible": completion >= float(cast(float, control["minimum_completion_rate"])),
        "mean_steps": report.mean_steps,
        "mean_total_model_input_tokens": report.metric_mean(METRIC_TOTAL_MODEL_INPUT_TOKENS),
        "statuses": [row["status"] for row in report.rows],
        "episode_diagnostics": [
            {
                "status": episode.status,
                "success": episode.success,
                "answer": episode.answer,
                "calls": [
                    {"name": name, "arguments": arguments}
                    for name, arguments, _observation in episode.transcript
                ],
            }
            for episode in report.episodes
        ],
    }
    return report, row


def run_transfer_matrix(
    design: dict[str, object], *, model: str, backend: str, complete: LLMComplete, data_dir: Path
) -> list[dict[str, object]]:
    """Run every predeclared depth/trigger cell after the pilot passes."""
    matrix = cast(dict[str, object], design["matrix"])
    rows: list[dict[str, object]] = []
    for depth in cast(list[int], matrix["depths"]):
        for share in cast(list[float], matrix["compact_shares"]):
            rows.append(
                run_transfer_cell(
                    model=model,
                    backend=backend,
                    complete=complete,
                    data_dir=data_dir,
                    n_tasks=int(cast(int, matrix["n_tasks"])),
                    depth=depth,
                    pad_chars=int(cast(int, matrix["pad_chars"])),
                    compact_share=share,
                    max_prompt_chars=int(cast(int, matrix["max_prompt_chars"])),
                    max_steps_margin=int(cast(int, matrix["max_steps_margin"])),
                    observation_cap_chars=int(cast(int, matrix["observation_cap_chars"])),
                    observation_head_share=float(cast(float, matrix["observation_head_share"])),
                    minimum_compaction_rate=float(cast(float, matrix["minimum_compaction_rate"])),
                    summary_input_cap=held_summary_input_cap(matrix),
                )
            )
    return rows


def analyze_transfer(
    design: dict[str, object],
    pilot_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Classify cross-family portability without changing a shipped default."""
    selected = next((row for row in pilot_rows if row["eligible"]), None)
    expected_cells = len(
        cast(list[int], cast(dict[str, object], design["matrix"])["depths"])
    ) * len(cast(list[float], cast(dict[str, object], design["matrix"])["compact_shares"]))
    if selected is None:
        reading = READING_INELIGIBLE
        reason = "no predeclared non-Qwen candidate passed the token-chain control pilot"
    elif len(matrix_rows) != expected_cells:
        raise ValueError(f"transfer matrix needs exactly {expected_cells} cells")
    else:
        prefer_compact = sum(row["verdict"] == VERDICT_PREFER_COMPACT for row in matrix_rows)
        prefer_cap = sum(row["verdict"] == VERDICT_PREFER_CAP for row in matrix_rows)
        if prefer_compact == expected_cells:
            reading = READING_PORTABLE
            reason = "compact is preferred in every eligible depth/trigger cell"
        elif prefer_compact:
            reading = READING_CONDITIONAL
            reason = (
                "compact transfers to the non-Qwen family, but its preference changes with geometry"
            )
        else:
            reading = READING_NOT_REPRODUCED
            reason = "no eligible non-Qwen matrix cell reproduces the compact preference"
        reason += f" ({prefer_compact} compact, {prefer_cap} cap, {expected_cells - prefer_compact - prefer_cap} other)"
    return {
        "study_id": design["study_id"],
        "selected_candidate": selected,
        "control_pilots": pilot_rows,
        "matrix_rows": matrix_rows,
        "transfer_reading": reading,
        "reason": reason,
        "changes_shipped_default": False,
    }
