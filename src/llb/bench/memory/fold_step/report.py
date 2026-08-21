"""Rendering and persistence for the fold-step crossover study."""

import json
from pathlib import Path
from typing import cast

from llb.bench.memory.fold_step.reading import (
    METHOD,
    READING_CONFIRMED,
    STUDY_KIND,
)
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_fold_step_table(analysis: dict[str, object]) -> str:
    """Render the control recheck, every cell against its declared step, and each depth's ladder."""
    lines = ["control recheck"]
    control = cast(dict[str, object] | None, analysis["control_recheck"])
    if control is None:
        lines.append("(not run)")
    else:
        lines.append(
            f"{control['model']}: completion={cast(float, control['completion']):.3f} "
            f"required={cast(float, control['minimum_completion_rate']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
    lines.extend(_cell_lines(cast(list[dict[str, object]], analysis["cells"])))
    for row in cast(list[dict[str, object]], analysis["depth_ladders"]):
        lines.extend(_ladder_lines(row))
    rule = cast(list[str], analysis["fold_step_routing_rule"])
    lines.extend(["", "fold-step routing rule"])
    lines.extend(f"- {line}" for line in rule or ["(not stated: no depth confirmed a boundary)"])
    lines.extend(
        [
            "",
            f"fold-step reading: [{analysis['fold_step_reading']}] {analysis['reason']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def _cell_lines(cells: list[dict[str, object]]) -> list[str]:
    if not cells:
        return []
    header = (
        f"{'cell':<26} {'depth':>5} {'guard':>6} {'trigger':>7} {'fold':>4} {'cap tok':>9} "
        f"{'compact tok':>11} {'d(tok)':>10} {'active':>6} {'side':<16} valid"
    )
    lines = ["", "cells", header, "-" * len(header)]
    for cell in cells:
        evidence = cast(dict[str, object], cell["cost_evidence"])
        delta = cast(dict[str, float], evidence["compact_minus_cap_total_model_input_tokens"])
        lines.append(
            f"{cast(str, cell['cell_id']):<26} {cast(int, cell['depth']):>5d} "
            f"{cast(int, cell['max_prompt_chars']):>6d} "
            f"{cast(int, cell['compaction_trigger_chars']):>7d} "
            f"{cast(int, cell['predicted_fold_step']):>4d} "
            f"{cast(float, cell['cap_mean_total_model_input_tokens']):>9.1f} "
            f"{cast(float, cell['compact_mean_total_model_input_tokens']):>11.1f} "
            f"{delta['mean']:>+10.1f} "
            f"{cast(float, cell['compaction_activation_rate']):>6.3f} "
            f"{cast(str, cell['measured_side']):<16} {str(cell['valid']).lower()}"
        )
    return lines


def _ladder_lines(row: dict[str, object]) -> list[str]:
    depth = cast(int, row["depth"])
    header = (
        f"{'fold step':<9} {'trigger interval':<20} {'guard interval':<20} {'guards':<16} "
        f"{'mean d(tok)':>12} {'spread':>8} {'ctrl':>6} {'summ':>6} {'band':>8} {'within':>6} side"
    )
    lines = [
        "",
        f"depth {depth} ladder (cap peak {row['cap_peak_prompt_chars']} chars, "
        f"prompt sequence {row['cap_prompt_sequence']})",
        header,
        "-" * len(header),
    ]
    for step in cast(list[dict[str, object]], row["steps"]):
        lines.append(
            f"{cast(int, step['fold_step']):<9d} "
            f"{_half_open(step['trigger_interval']):<20} "
            f"{_half_open(step['guard_interval']):<20} "
            f"{str(step['guards']):<16} {cast(float, step['mean_cost_delta']):>+12.1f} "
            f"{cast(float, step['spread']):>8.1f} "
            f"{_optional(step['controller_prompt_spread']):>6} "
            f"{_optional(step['summarizer_prompt_spread']):>6} "
            f"{cast(float, step['equivalence_band']):>8.1f} "
            f"{str(step['within_band']).lower():>6} {step['side']}"
        )
    boundary = cast(dict[str, object] | None, row["boundary"])
    if boundary is not None:
        lines.append(
            f"boundary: last compact-cheaper fold step {row['last_compact_cheaper_fold_step']} "
            f"(trigger < {boundary['trigger_boundary_chars']} chars, guard < "
            f"{boundary['guard_boundary_chars']} chars); the "
            f"{boundary['from_fold_step']}->{boundary['to_fold_step']} change is straddled by "
            f"{boundary['straddling_guard_gap_chars']} chars of guard and moves "
            f"{cast(float, boundary['step_change_separation']):.1f} tokens against a "
            f"{cast(float, boundary['step_change_band']):.1f}-token band"
        )
    artifact = cast(dict[str, object] | None, row["interpolated_guard_artifact"])
    if artifact is not None:
        lines.append(
            f"interpolated guard {artifact['guard_chars']} folds at step {artifact['fold_step']} "
            f"inside {artifact['guard_interval']}, {artifact['gap_to_boundary_chars']} chars below "
            "that step's own change"
        )
    lines.append(
        f"depth {depth} reading: [{row['reading']}] "
        f"controller cost is an exact step function: "
        f"{str(row['controller_cost_is_exact_step']).lower()}, within-step residual "
        f"{cast(float, row['within_step_residual_tokens']):.1f} tokens"
    )
    return lines


def _half_open(interval: object) -> str:
    """Render `[low, high)` so the exclusive upper end -- the step change itself -- is unambiguous."""
    low, high = cast(list[int], interval)
    return f"[{low}, {high})"


def _optional(value: object) -> str:
    """A recorded spread, or `-` for a bundle that predates the controller/summarizer split."""
    return "-" if value is None else f"{cast(float, value):.1f}"


def persist_fold_steps(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the aggregate, the immutable design, and the rendered ladders."""
    cells = cast(list[dict[str, object]], analysis["cells"])
    steps = [
        step
        for row in cast(list[dict[str, object]], analysis["depth_ladders"])
        for step in cast(list[dict[str, object]], row["steps"])
    ]
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=cast(str, design["study_id"]),
        config={
            "category": METHOD,
            "study_kind": STUDY_KIND,
            "design": design,
            "analysis": analysis,
        },
        metrics={
            "objective_score": 1.0 if analysis["fold_step_reading"] == READING_CONFIRMED else 0.0,
            "reliability": (
                sum(bool(step["within_band"] and step["same_side"]) for step in steps) / len(steps)
                if steps
                else 0.0
            ),
            "tokens_per_s": tokens_per_s,
        },
        case_rows=cells,
        mirror=mirror,
        artifacts={
            "fold-step-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "fold-step-analysis.json": json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            "fold-step-crossover.md": (
                "# Compact fold-step crossover\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
