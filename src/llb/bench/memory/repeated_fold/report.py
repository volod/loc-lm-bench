"""Rendering and persistence for repeated-fold completion evidence."""

import json
from pathlib import Path
from typing import cast

from llb.bench.context_policy.compact_vs_cap_report import METHOD
from llb.bench.memory.repeated_fold.completion import RepeatedFoldRun
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_repeated_fold_table(analysis: dict[str, object]) -> str:
    """Render both mechanism arms and the completion rate at each measured fold count."""
    header = f"{'cell':<24} {'arm':<18} {'guard':>6} {'oracle':>6} {'folds':<20} {'complete':>8}"
    lines = ["cells", header, "-" * len(header)]
    for row in cast(list[dict[str, object]], analysis["cells"]):
        folds = ",".join(str(value) for value in cast(list[int], row["measured_fold_counts"]))
        lines.append(
            f"{cast(str, row['cell_id']):<24} {cast(str, row['arm']):<18} "
            f"{cast(int, row['max_prompt_chars']):>6d} "
            f"{cast(int, row['oracle_folds']):>6d} {folds:<20} "
            f"{cast(float, row['completion']):>8.3f}"
        )
    lines.extend(["", "completion by measured fold count"])
    for row in cast(list[dict[str, object]], analysis["completion_by_measured_fold_count"]):
        lines.append(
            f"- folds={row['measured_folds']}: {row['n_completed']}/{row['n_cases']} "
            f"({cast(float, row['completion']):.3f})"
        )
    lines.extend(
        [
            "",
            f"completion reading: [{analysis['completion_reading']}] "
            f"{analysis['completion_reason']}",
            f"recommended fold-count limit: {analysis['recommended_fold_count_limit']}",
            f"mechanism reading: [{analysis['mechanism_reading']}] {analysis['mechanism_reason']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def persist_repeated_fold_cells(
    design: dict[str, object],
    run: RepeatedFoldRun,
    *,
    data_dir: Path | str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
    run_prefix: str = "repeated-fold",
) -> list[dict[str, object]]:
    """Persist one bundle per (cell, marker arm) and stamp each row with its manifest path.

    `run_prefix` is what keeps a replication's per-family bundles addressable: the same cells run
    once per model family, so the family has to be in the run name or the second family's bundles
    are indistinguishable from the first's.
    """
    rows = cast(list[dict[str, object]], run.analysis["cells"])
    for row in rows:
        key = (cast(str, row["cell_id"]), cast(str, row["arm"]))
        report = run.reports[key]
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=f"{run_prefix}-{key[0]}-{key[1]}",
            config={
                "category": METHOD,
                "study_id": design["study_id"],
                "study_kind": design["study_kind"],
                "model": run.analysis["model"],
                "backend": run.analysis["backend"],
                "seed": design["seed"],
                "task_set_digest": run.analysis["task_set_digest"],
                "cell": row,
            },
            metrics={
                "objective_score": report.result.objective_score,
                "reliability": report.reliability,
                "tokens_per_s": tokens_per_s,
            },
            case_rows=cast(list[dict[str, object]], row["cases"]),
            mirror=mirror,
        )
        row["manifest"] = str(paths["manifest"])
    return rows


def mean_shipped_completion(rows: list[dict[str, object]]) -> float:
    """Mean completion over the shipped typed-marker arm: the aggregate's objective score."""
    typed = [row for row in rows if row["arm"] == "typed_marker"]
    return sum(cast(float, row["completion"]) for row in typed) / len(typed)


def persist_repeated_fold_run(
    design: dict[str, object],
    run: RepeatedFoldRun,
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist each compact arm and one aggregate under the established method root."""
    rows = persist_repeated_fold_cells(
        design, run, data_dir=data_dir, tokens_per_s=tokens_per_s, mirror=mirror
    )
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=cast(str, design["study_id"]),
        config={
            "category": METHOD,
            "study_id": design["study_id"],
            "study_kind": design["study_kind"],
            "design": design,
            "analysis": run.analysis,
        },
        metrics={
            "objective_score": mean_shipped_completion(rows),
            "reliability": 1.0,
            "tokens_per_s": tokens_per_s,
        },
        case_rows=rows,
        mirror=mirror,
        study_id=cast(str, design["study_id"]),
        artifacts={
            "repeated-fold-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "repeated-fold-analysis.json": json.dumps(run.analysis, indent=2, sort_keys=True)
            + "\n",
            "repeated-fold-completion.md": "# Repeated-fold completion\n\n```text\n"
            + table
            + "\n```\n",
        },
    )
