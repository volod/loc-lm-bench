"""Aggregate rendering and persistence for compact-memory transfer."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_transfer import METHOD, STUDY_KIND
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_transfer_table(analysis: dict[str, object]) -> str:
    """Render pilot eligibility and the paired completion/cost matrix."""
    lines = ["control pilots"]
    for row in cast(list[dict[str, object]], analysis["control_pilots"]):
        lines.append(
            f"{row['model']}: completion={cast(float, row['completion']):.3f} "
            f"required={cast(float, row['minimum_completion_rate']):.3f} "
            f"eligible={str(row['eligible']).lower()}"
        )
    matrix = cast(list[dict[str, object]], analysis["matrix_rows"])
    if matrix:
        header = (
            f"{'depth':>5} {'trigger':>7} {'cap':>7} {'compact':>8} {'d(complete)':>11} "
            f"{'cap tok':>9} {'compact tok':>11} {'d(tok)':>10} {'active':>7} verdict"
        )
        lines.extend(["", "depth/trigger matrix", header, "-" * len(header)])
        for row in matrix:
            paired = cast(dict[str, dict[str, object]], row["paired"])
            completion = cast(dict[str, float], paired["completion"]["delta"])
            cost = cast(dict[str, float], paired["total_model_input_tokens"]["delta"])
            lines.append(
                f"{cast(int, row['depth']):>5d} {cast(float, row['compact_share']):>7.2f} "
                f"{cast(float, row['cap_completion']):>7.3f} "
                f"{cast(float, row['compact_completion']):>8.3f} "
                f"{completion['mean']:>+11.3f} "
                f"{cast(float, row['cap_mean_total_model_input_tokens']):>9.1f} "
                f"{cast(float, row['compact_mean_total_model_input_tokens']):>11.1f} "
                f"{cost['mean']:>+10.1f} "
                f"{cast(float, row['compaction_activation_rate']):>7.3f} {row['verdict']}"
            )
    lines.extend(
        [
            "",
            f"transfer reading: [{analysis['transfer_reading']}] {analysis['reason']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def persist_transfer(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist one self-contained aggregate linked to every policy cell."""
    selected = cast(dict[str, object] | None, analysis["selected_candidate"])
    matrix = cast(list[dict[str, object]], analysis["matrix_rows"])
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
            "objective_score": (
                sum(
                    cast(float, row["compact_completion"]) > cast(float, row["cap_completion"])
                    for row in matrix
                )
                / len(matrix)
                if matrix
                else 0.0
            ),
            "reliability": 1.0 if selected is not None else 0.0,
            "tokens_per_s": tokens_per_s,
        },
        case_rows=matrix,
        mirror=mirror,
        artifacts={
            "transfer-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "transfer-analysis.json": json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            "transfer-comparison.md": (
                "# Compact-memory cross-model transfer\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
