"""Rendering and persistence for family-adapted repeat-feedback evidence."""

import json
from pathlib import Path
from typing import cast

from llb.bench.loop_feedback.adaptation_design import STUDY_KIND
from llb.bench.loop_policy.report import METHOD
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_feedback_adaptation_table(analysis: dict[str, object]) -> str:
    """Render the seeded gates and stable family-specific routes."""
    header = (
        f"{'family':<8} {'seed':>5} {'candidate':<14} {'eligible':<8} {'gates':<5} "
        f"{'response':>8} "
        f"{'complete':>8} {'d(complete)':>11} {'d(prompt)':>10} {'d(wall)':>8} supports"
    )
    lines = [header, "-" * len(header)]
    for row in cast(list[dict[str, object]], analysis["seed_rows"]):
        lines.append(
            f"{cast(str, row['model_family']):<8} {cast(int, row['seed']):>5d} "
            f"{cast(str, row['candidate_feedback_variant']):<14} "
            f"{str(row['eligible']).lower():<8} "
            f"{'C' if row['completion_gate_passed'] else '-'}"
            f"{'P' if row['prompt_cost_gate_passed'] else '-'}"
            f"{'W' if row['wall_cost_gate_passed'] else '-':<3} "
            f"{cast(float, row['response_rate']):>8.3f} "
            f"{cast(float, row['completion_rate']):>8.3f} "
            f"{cast(float, row['completion_delta']):>+11.3f} "
            f"{cast(float, row['prompt_token_delta']):>+10.1f} "
            f"{cast(float, row['wall_clock_delta_s']):>+8.2f} "
            f"{str(row['supports_candidate']).lower()}"
        )
    lines.extend(["", "stable family routing"])
    for family, row in cast(dict[str, dict[str, object]], analysis["families"]).items():
        lines.append(
            f"{family}: {row['supported_seeds']}/{row['required_supported_seeds']} seeds -> "
            f"{row['routed_feedback_variant']}"
        )
    return "\n".join(lines)


def persist_feedback_adaptation(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    task_digest: str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the aggregate routing decision beside its source policy cells."""
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=cast(str, design["study_id"]),
        config={
            "category": METHOD,
            "study_kind": STUDY_KIND,
            "task_set_digest": task_digest,
            "design": design,
            "analysis": analysis,
        },
        metrics={
            "objective_score": float(cast(float, analysis["supported_family_fraction"])),
            "reliability": 1.0 if analysis["coverage_and_activation_passed"] else 0.0,
            "tokens_per_s": 0.0,
        },
        case_rows=cast(list[dict[str, object]], analysis["seed_rows"]),
        mirror=mirror,
        artifacts={
            "family-adaptation-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "family-adaptation-analysis.json": (
                json.dumps(analysis, indent=2, sort_keys=True) + "\n"
            ),
            "family-adaptation-comparison.md": (
                "# Repeat-feedback family adaptation\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
