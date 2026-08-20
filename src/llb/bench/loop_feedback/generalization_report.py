"""Rendering and persistence for repeat-feedback generalization evidence."""

import json
from pathlib import Path
from typing import cast

from llb.bench.loop_feedback.generalization import STUDY_KIND
from llb.bench.loop_policy.report import METHOD
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_feedback_generalization_table(analysis: dict[str, object]) -> str:
    """Render family/seed evidence and the stable family routing decision."""
    header = (
        f"{'family':<10} {'seed':>5} {'eligible':<8} {'response':>8} {'complete':>8} "
        f"{'d(complete)':>11} {'d(prompt)':>10} {'d(wall)':>8} supports"
    )
    lines = [header, "-" * len(header)]
    for row in cast(list[dict[str, object]], analysis["seed_rows"]):
        lines.append(
            f"{cast(str, row['model_family']):<10} {cast(int, row['seed']):>5d} "
            f"{str(row['eligible']).lower():<8} {cast(float, row['response_rate']):>8.3f} "
            f"{cast(float, row['completion_rate']):>8.3f} "
            f"{cast(float, row['completion_delta']):>+11.3f} "
            f"{cast(float, row['prompt_token_delta']):>+10.1f} "
            f"{cast(float, row['wall_clock_delta_s']):>+8.2f} "
            f"{str(row['supports_bilingual']).lower()}"
        )
    lines.extend(["", "family routing"])
    for family, row in cast(dict[str, dict[str, object]], analysis["families"]).items():
        lines.append(
            f"{family}: {row['supported_seeds']}/{len(cast(list[int], row['seeds']))} seeds -> "
            f"{row['routed_feedback_variant']}"
        )
    return "\n".join(lines)


def persist_feedback_generalization(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    task_digest: str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist one aggregate bundle beside its source agent-loop-policy cells."""
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
            "generalization-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "generalization-analysis.json": json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            "generalization-comparison.md": (
                "# Repeat-feedback generalization\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
