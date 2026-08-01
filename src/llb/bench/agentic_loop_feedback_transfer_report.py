"""Rendering and persistence for Gemma task-family feedback transfer."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_loop_policy_report import METHOD
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_feedback_transfer_table(analysis: dict[str, object]) -> str:
    """Render seeded transfer gates and compact per-family response rates."""
    header = (
        f"{'seed':>5} {'eligible':<8} {'gates':<6} {'base-r':>7} {'cand-r':>7} "
        f"{'complete':>8} {'d(complete)':>11} {'d(prompt)':>10} {'d(wall)':>8} "
        "family-response/completion"
    )
    lines = [header, "-" * len(header)]
    for row in cast(list[dict[str, object]], analysis["seed_rows"]):
        family_outcomes = cast(dict[str, dict[str, float]], row["task_family_response_completion"])
        response_text = ",".join(
            f"{family.removesuffix('_holdout')}={outcome['response_rate']:.3f}/"
            f"{outcome['redirected_completion_rate']:.3f}"
            for family, outcome in family_outcomes.items()
        )
        gates = (
            f"{'R' if row['task_family_response_gate_passed'] else '-'}"
            f"{'C' if row['completion_gate_passed'] else '-'}"
            f"{'P' if row['prompt_cost_gate_passed'] else '-'}"
            f"{'W' if row['wall_cost_gate_passed'] else '-'}"
        )
        lines.append(
            f"{cast(int, row['seed']):>5d} {str(row['eligible']).lower():<8} "
            f"{gates:<6} {cast(float, row['baseline_response_rate']):>7.3f} "
            f"{cast(float, row['response_rate']):>7.3f} "
            f"{cast(float, row['completion_rate']):>8.3f} "
            f"{cast(float, row['completion_delta']):>+11.3f} "
            f"{cast(float, row['prompt_token_delta']):>+10.1f} "
            f"{cast(float, row['wall_clock_delta_s']):>+8.2f} {response_text}"
        )
    return "\n".join(lines)


def persist_feedback_transfer(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    task_digest: str,
    table: str,
    mirror: Mirror | None = None,
    artifact_stem: str = "task-family-transfer",
    report_title: str = "Gemma repeat-feedback task-family transfer",
) -> RunPaths:
    """Persist the two-seed decision beside every source policy-cell manifest."""
    supported = int(cast(int, analysis["supported_seeds"]))
    required = int(cast(int, analysis["required_supported_seeds"]))
    eligible = all(
        bool(row["eligible"]) for row in cast(list[dict[str, object]], analysis["seed_rows"])
    )
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=cast(str, design["study_id"]),
        config={
            "category": METHOD,
            "study_kind": design["study_kind"],
            "task_set_digest": task_digest,
            "design": design,
            "analysis": analysis,
        },
        metrics={
            "objective_score": supported / required,
            "reliability": 1.0 if eligible else 0.0,
            "tokens_per_s": 0.0,
        },
        case_rows=cast(list[dict[str, object]], analysis["seed_rows"]),
        mirror=mirror,
        artifacts={
            f"{artifact_stem}-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            f"{artifact_stem}-analysis.json": (
                json.dumps(analysis, indent=2, sort_keys=True) + "\n"
            ),
            f"{artifact_stem}-comparison.md": (
                f"# {report_title}\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
