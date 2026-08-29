"""Rendering and persistence for the two-family repeated-fold replication."""

import json
from pathlib import Path
from typing import cast

from llb.bench.context_policy.compact_vs_cap_report import METHOD
from llb.bench.memory.repeated_fold.replication import ReplicationFamilyRun
from llb.bench.memory.repeated_fold.report import (
    mean_shipped_completion,
    persist_repeated_fold_cells,
)
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_replication_table(analysis: dict[str, object]) -> str:
    """Render every family's measured fold groups, their intervals, and the cross-family rule."""
    lines: list[str] = []
    for family in cast(list[dict[str, object]], analysis["families"]):
        lines.extend(_family_block(family))
        lines.append("")
    lines.extend(
        [
            f"task-set digest: {analysis['task_set_digest']}",
            f"family digest: {analysis['family_digest']} (roster {analysis['roster_digest']})",
            f"qualified families: {analysis['qualified_models']}",
            f"shared powered fold limit: {analysis['shared_powered_fold_limit']}",
            f"replication reading: [{analysis['replication_reading']}] "
            f"{analysis['replication_reason']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def _family_block(family: dict[str, object]) -> list[str]:
    header = (
        f"{'folds':>5} {'evidence':>9} {'kind':<14} {'complete':>8} {'95% ci':<16} "
        f"{'pairs':>5} {'ctrl-w':>6} {'grp-w':>5} {'marker-w':>8}"
    )
    lines = [
        f"family={family['model_family']} model={family['model']} "
        f"eligible={str(family['control_eligible']).lower()} "
        f"powered-fold-limit={family['powered_fold_limit']}",
        header,
        "-" * len(header),
    ]
    for row in cast(list[dict[str, object]], family["fold_groups"]):
        paired = cast(dict[str, object], row["paired"])
        ablation = cast(dict[str, int], row["marker_ablation"])
        interval = (
            f"[{cast(float, row['completion_lo']):.3f}, {cast(float, row['completion_hi']):.3f}]"
        )
        floor = "" if row["meets_evidence_floor"] else " *under-floor"
        lines.append(
            f"{cast(int, row['measured_folds']):>5d} {cast(int, row['n_evidence']):>9d} "
            f"{cast(str, row['evidence_kind']):<14} "
            f"{cast(float, row['completion']):>8.3f} {interval:<16} "
            f"{cast(int, paired['n_pairs']):>5d} {cast(int, paired['control_wins']):>6d} "
            f"{cast(int, paired['group_wins']):>5d} {ablation['marker_wins']:>8d}{floor}"
        )
    lines.append(f"  powered-fold reason: {family['powered_fold_reason']}")
    lines.append(f"  mechanism: [{family['mechanism_reading']}] {family['mechanism_reason']}")
    return lines


def persist_replication_run(
    design: dict[str, object],
    runs: list[ReplicationFamilyRun],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist every family's cell bundles and one cross-family aggregate."""
    completions: list[float] = []
    for run in runs:
        rows = persist_repeated_fold_cells(
            design,
            run.base,
            data_dir=data_dir,
            tokens_per_s=run.tokens_per_s,
            mirror=mirror,
            run_prefix=f"repeated-fold-replication-{run.model_family}",
        )
        completions.append(mean_shipped_completion(rows))
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=cast(str, design["study_id"]),
        config={
            "category": METHOD,
            "study_id": design["study_id"],
            "study_kind": design["study_kind"],
            "design": design,
            "analysis": analysis,
        },
        metrics={
            "objective_score": sum(completions) / len(completions) if completions else 0.0,
            "reliability": 1.0,
            "tokens_per_s": max((run.tokens_per_s for run in runs), default=0.0),
        },
        case_rows=[
            {
                "model_family": run.model_family,
                "model": run.model,
                "backend": run.backend,
                "tokens_per_s": run.tokens_per_s,
                "control_eligible": run.analysis["control_eligible"],
                "powered_fold_limit": run.analysis["powered_fold_limit"],
                "task_set_digest": run.analysis["task_set_digest"],
            }
            for run in runs
        ],
        mirror=mirror,
        artifacts={
            "repeated-fold-replication-design.json": json.dumps(design, indent=2, sort_keys=True)
            + "\n",
            "repeated-fold-replication-analysis.json": json.dumps(
                analysis, indent=2, sort_keys=True
            )
            + "\n",
            "repeated-fold-replication.md": "# Repeated-fold completion replication\n\n```text\n"
            + table
            + "\n```\n",
        },
    )
