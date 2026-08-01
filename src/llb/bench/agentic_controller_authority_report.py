"""Persistence and rendering for controller-channel authority evidence."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_controller_authority import ChannelCell
from llb.bench.agentic.model import STATUS_COMPLETED
from llb.bench.agentic_loop_policy_report import METHOD
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def persist_channel_cell(
    design: dict[str, object],
    cell: ChannelCell,
    *,
    seed: int,
    model: str,
    backend: str,
    data_dir: Path | str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist one placement cell, including its exact first-authority prompt snapshots."""
    completion = sum(float(row["success"]) for row in cell.rows) / len(cell.rows)
    reliability = sum(row["status"] == STATUS_COMPLETED for row in cell.rows) / len(cell.rows)
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=f"{design['study_id']}-seed={seed}-placement={cell.placement}",
        config={
            "category": METHOD,
            "study_kind": design["study_kind"],
            "model": model,
            "backend": backend,
            "seed": seed,
            "placement": cell.placement,
            "role_serialization": design["role_serialization"],
            "task_set_digest": cast(dict[str, object], design["reference"])["task_set_digest"],
        },
        metrics={
            "objective_score": completion,
            "reliability": reliability,
            "tokens_per_s": cell.tokens_per_s,
        },
        case_rows=cell.rows,
        mirror=mirror,
        artifacts={
            "prompt-snapshots.json": json.dumps(
                cell.snapshots, ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
        },
    )


def format_channel_authority_table(analysis: dict[str, object]) -> str:
    """Render both seeded controller-channel gate decisions."""
    header = (
        f"{'seed':>5} {'gates':<6} {'obs-r':>7} {'ctrl-r':>7} {'obs-c':>7} {'ctrl-c':>7} "
        f"{'d(comp)':>8} {'d(prompt)':>10} {'d(wall)':>8} family-response/completion"
    )
    lines = [header, "-" * len(header)]
    for row in cast(list[dict[str, object]], analysis["seed_rows"]):
        completion = cast(dict[str, object], row["completion_comparison"])
        completion_delta = cast(dict[str, float], completion["delta"])["mean"]
        cost = cast(dict[str, dict[str, object]], row["cost"])
        prompt_delta = cast(dict[str, float], cost["total_model_input_tokens"]["paired_delta"])[
            "mean"
        ]
        wall_delta = cast(dict[str, float], cost["elapsed_s"]["paired_delta"])["mean"]
        families = cast(dict[str, dict[str, float]], row["task_family_response_completion"])
        family_text = ",".join(
            f"{name.removesuffix('_holdout')}={values['response_rate']:.3f}/"
            f"{values['redirected_completion_rate']:.3f}"
            for name, values in families.items()
        )
        gates = (
            f"{'S' if cast(dict[str, object], row['snapshot_proof'])['passed'] else '-'}"
            f"{'A' if row['activation_passed'] else '-'}"
            f"{'R' if row['task_family_response_gate_passed'] else '-'}"
            f"{'C' if row['completion_gate_passed'] else '-'}"
            f"{'P' if cost['total_model_input_tokens']['passed'] else '-'}"
            f"{'W' if cost['elapsed_s']['passed'] else '-'}"
        )
        lines.append(
            f"{cast(int, row['seed']):>5d} {gates:<6} "
            f"{cast(float, row['baseline_response_rate']):>7.3f} "
            f"{cast(float, row['response_rate']):>7.3f} "
            f"{cast(float, row['baseline_completion_rate']):>7.3f} "
            f"{cast(float, row['completion_rate']):>7.3f} "
            f"{completion_delta:>+8.3f} {prompt_delta:>+10.1f} {wall_delta:>+8.2f} "
            f"{family_text}"
        )
    return "\n".join(lines)


def persist_channel_authority(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the final two-seed structural-authority decision."""
    supported = int(cast(int, analysis["supported_seeds"]))
    required = int(cast(int, analysis["required_supported_seeds"]))
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=cast(str, design["study_id"]),
        config={
            "category": METHOD,
            "study_kind": design["study_kind"],
            "design": design,
            "analysis": analysis,
        },
        metrics={
            "objective_score": supported / required,
            "reliability": 1.0,
            "tokens_per_s": 0.0,
        },
        case_rows=cast(list[dict[str, object]], analysis["seed_rows"]),
        mirror=mirror,
        artifacts={
            "controller-channel-authority-design.json": (
                json.dumps(design, indent=2, sort_keys=True) + "\n"
            ),
            "controller-channel-authority-analysis.json": (
                json.dumps(analysis, indent=2, sort_keys=True) + "\n"
            ),
            "controller-channel-authority-comparison.md": (
                "# Controller-channel authority comparison\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
