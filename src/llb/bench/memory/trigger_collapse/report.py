"""Rendering and persistence for the trigger/guard collapse study."""

from pathlib import Path
from typing import cast

from llb.artifacts.runs.members import study_analysis, study_design, table_report
from llb.bench.memory.fold_step.ladder import compaction_trigger_chars
from llb.bench.memory.trigger_collapse.reading import (
    KIND_EQUAL_TRIGGER,
    METHOD,
    READING_COLLAPSES,
    STUDY_KIND,
)
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_collapse_table(analysis: dict[str, object]) -> str:
    """Render the control recheck, every cell's trigger and cost, and the family bands."""
    control = cast(dict[str, object] | None, analysis["control_recheck"])
    lines = ["control recheck"]
    if control is None:
        lines.append("(not run)")
    else:
        lines.append(
            f"{control['model']}: completion={cast(float, control['completion']):.3f} "
            f"required={cast(float, control['minimum_completion_rate']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
    cells = cast(list[dict[str, object]], analysis["cells"])
    if cells:
        header = (
            f"{'cell':<26} {'depth':>5} {'share':>5} {'guard':>6} {'trigger':>7} {'fold':>4} "
            f"{'cap tok':>9} {'compact tok':>11} {'d(tok)':>10} {'active':>6} {'side':<16} valid"
        )
        lines.extend(["", "cells", header, "-" * len(header)])
        for cell in cells:
            evidence = cast(dict[str, object], cell["cost_evidence"])
            delta = cast(dict[str, float], evidence["compact_minus_cap_total_model_input_tokens"])
            guard = cast(int, cell["max_prompt_chars"])
            share = cast(float, cell["compact_share"])
            fold = cell.get("predicted_fold_step")
            lines.append(
                f"{cast(str, cell['cell_id']):<26} "
                f"{cast(int, cell['depth']):>5d} {share:>5.2f} {guard:>6d} "
                f"{compaction_trigger_chars(guard, share):>7d} "
                f"{'-' if fold is None else cast(int, fold):>4} "
                f"{cast(float, cell['cap_mean_total_model_input_tokens']):>9.1f} "
                f"{cast(float, cell['compact_mean_total_model_input_tokens']):>11.1f} "
                f"{delta['mean']:>+10.1f} "
                f"{cast(float, cell['compaction_activation_rate']):>6.3f} "
                f"{cast(str, cell['measured_side']):<16} {str(cell['valid']).lower()}"
            )
    families = cast(list[dict[str, object]], analysis["families"])
    if families:
        header = (
            f"{'family':<20} {'kind':<14} {'depth':>5} {'triggers':<18} {'folds':<10} "
            f"{'spread':>9} {'band':>8} {'within':>6} same-side"
        )
        lines.extend(["", "families", header, "-" * len(header)])
        for row in families:
            lines.append(
                f"{cast(str, row['family_id']):<20} {cast(str, row['kind']):<14} "
                f"{cast(int, row['depth']):>5d} "
                f"{str(row['triggers']):<18} "
                f"{str(row.get('fold_steps', [])):<10} "
                f"{cast(float, row['spread']):>9.1f} "
                f"{cast(float, row['equivalence_band']):>8.1f} "
                f"{str(row['within_band']).lower():>6} {str(row['same_side']).lower()}"
            )
    rule = cast(list[str], analysis["portable_trigger_rule"])
    lines.extend(["", "portable trigger rule"])
    lines.extend(f"- {line}" for line in rule or ["(not stated: the collapse did not hold)"])
    lines.extend(
        [
            "",
            f"collapse reading: [{analysis['collapse_reading']}] {analysis['reason']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def persist_collapse(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the aggregate, the immutable design, and the rendered families."""
    cells = cast(list[dict[str, object]], analysis["cells"])
    families = cast(list[dict[str, object]], analysis["families"])
    held = [row for row in families if row["kind"] == KIND_EQUAL_TRIGGER]
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
            "objective_score": 1.0 if analysis["collapse_reading"] == READING_COLLAPSES else 0.0,
            "reliability": (
                sum(bool(row["within_band"] and row["same_side"]) for row in held) / len(held)
                if held
                else 0.0
            ),
            "tokens_per_s": tokens_per_s,
        },
        case_rows=cells,
        mirror=mirror,
        artifacts=[
            study_design("trigger-collapse-design.json", design),
            study_analysis("trigger-collapse-analysis.json", analysis),
            table_report("trigger-collapse.md", "Compact trigger-versus-guard collapse", table),
        ],
    )
