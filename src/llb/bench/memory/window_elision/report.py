"""Rendering and persistence for window-bound summary elision evidence."""

import json
from pathlib import Path
from typing import cast

from llb.bench.memory.window_elision.run import WindowElisionRun
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths

METHOD = "agentic-compact-window-elision"


def format_window_elision_table(analysis: dict[str, object]) -> str:
    """Render the geometry, actual completion, and bounded recommendation."""
    header = f"{'cell':<28} {'role':<18} {'guard':>6} {'trigger':>7} {'input':>6} {'elided':>7} {'complete':>8}"
    lines = [header, "-" * len(header)]
    for row in cast(list[dict[str, object]], analysis["cells"]):
        lines.append(
            f"{cast(str, row['cell_id']):<28} {cast(str, row['role']):<18} "
            f"{cast(int, row['max_prompt_chars']):>6d} "
            f"{cast(int, row['compaction_trigger_chars']):>7d} "
            f"{cast(int, row['summary_input_chars']):>6d} "
            f"{cast(int, row['summary_input_elided_chars']):>7d} "
            f"{cast(float, row['completion']):>8.3f}"
        )
    paired = cast(dict[str, object], analysis["paired_completion"])
    lines.extend(
        [
            "",
            f"eligible: {str(analysis['comparison_eligible']).lower()} - {analysis['eligibility_reason']}",
            f"paired: fit wins={paired['fit_wins']}, elided wins={paired['elided_wins']}, "
            f"unchanged={paired['unchanged']}, delta={float(cast(float, paired['completion_delta'])):+.3f}",
            f"completion reading: [{analysis['completion_reading']}] {analysis['completion_reason']}",
            f"operator: {analysis['operator_recommendation']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def persist_window_elision_run(
    design: dict[str, object],
    run: WindowElisionRun,
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist both category arms and one aggregate with the exact design and analysis."""
    rows = cast(list[dict[str, object]], run.analysis["cells"])
    for row in rows:
        cell_id = cast(str, row["cell_id"])
        report = run.reports[cell_id]
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=cell_id,
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
    objective = sum(cast(float, row["completion"]) for row in rows) / len(rows)
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
        metrics={"objective_score": objective, "reliability": 1.0, "tokens_per_s": tokens_per_s},
        case_rows=rows,
        mirror=mirror,
        study_id=cast(str, design["study_id"]),
        artifacts={
            "window-elision-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "window-elision-analysis.json": json.dumps(run.analysis, indent=2, sort_keys=True)
            + "\n",
            "window-elision.md": "# Window-bound summary elision\n\n```text\n" + table + "\n```\n",
        },
    )
