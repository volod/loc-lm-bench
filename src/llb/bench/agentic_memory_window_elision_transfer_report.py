"""Rendering and persistence for stratum-controlled window-elision transfer."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_window_elision_transfer import TransferFamilyRun
from llb.bench.agentic_memory_window_elision_tasks import STRATA
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths

METHOD = "agentic-compact-window-elision-transfer"


def format_window_elision_transfer_table(analysis: dict[str, object]) -> str:
    """Render family qualification, per-stratum pairs, and conditional prototype."""
    header = f"{'family':<14} {'eligible':<8} {'stratum':<8} {'fit wins':>8} {'elided wins':>12} {'same':>5} {'reading'}"
    lines = [header, "-" * len(header)]
    for family in cast(list[dict[str, object]], analysis["families"]):
        strata = cast(dict[str, dict[str, object]], family["strata"])
        for stratum in STRATA:
            row = strata[stratum]
            paired = cast(dict[str, object], row["paired"])
            lines.append(
                f"{cast(str, family['model_family']):<14} "
                f"{str(family['eligible']).lower():<8} {stratum:<8} "
                f"{cast(int, paired['fit_wins']):>8d} "
                f"{cast(int, paired['elided_wins']):>12d} "
                f"{cast(int, paired['unchanged']):>5d} {row['reading']}"
            )
    lines.extend(
        [
            "",
            f"transfer: [{analysis['transfer_reading']}] {analysis['transfer_reason']}",
            f"prototype required: {str(analysis['prototype_required']).lower()}",
            f"prototype: [{analysis['prototype_reading']}] {analysis['prototype_reason']}",
            "shipped default changed: false",
        ]
    )
    prototype = cast(dict[str, object], analysis["prototype_detail"])
    family_rows = cast(list[dict[str, object]], prototype.get("families", []))
    if family_rows:
        lines.extend(["", "entry-aware minus head-tail prototype"])
        for family in family_rows:
            prototype_strata = cast(dict[str, dict[str, int]], family["strata"])
            for stratum in STRATA:
                prototype_row = prototype_strata[stratum]
                lines.append(
                    f"- model={family['model']} stratum={stratum}: "
                    f"entry-aware wins={prototype_row['entry_aware_wins']}, "
                    f"head-tail wins={prototype_row['head_tail_wins']}, "
                    f"unchanged={prototype_row['unchanged']}"
                )
            lines.append(f"  same summary prompt chars: {str(family['same_prompt_chars']).lower()}")
    return "\n".join(lines)


def persist_window_elision_transfer(
    design: dict[str, object],
    runs: list[TransferFamilyRun],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist each family/strategy cell and one cross-family aggregate."""
    persisted_rows: list[dict[str, object]] = []
    for run in runs:
        persisted_rows.extend(_persist_family_cells(run, design, data_dir=data_dir, mirror=mirror))
    objective = (
        sum(float(cast(float, row["completion"])) for row in persisted_rows) / len(persisted_rows)
        if persisted_rows
        else 0.0
    )
    throughputs = [run.tokens_per_s for run in runs if run.tokens_per_s > 0.0]
    throughputs.extend(
        run.prototype_tokens_per_s for run in runs if run.prototype_tokens_per_s > 0.0
    )
    tokens_per_s = sum(throughputs) / len(throughputs) if throughputs else 0.0
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
            "objective_score": objective,
            "reliability": 1.0,
            "tokens_per_s": tokens_per_s,
        },
        case_rows=persisted_rows,
        mirror=mirror,
        artifacts={
            "window-elision-transfer-design.json": json.dumps(design, indent=2, sort_keys=True)
            + "\n",
            "window-elision-transfer-analysis.json": json.dumps(analysis, indent=2, sort_keys=True)
            + "\n",
            "window-elision-transfer.md": "# Middle-critical window-elision transfer\n\n```text\n"
            + table
            + "\n```\n",
        },
    )


def _persist_family_cells(
    run: TransferFamilyRun,
    design: dict[str, object],
    *,
    data_dir: Path | str,
    mirror: Mirror | None,
) -> list[dict[str, object]]:
    rows = cast(list[dict[str, object]], run.base.analysis["cells"])
    reports = run.base.reports
    if run.prototype_row is not None and run.prototype_report is not None:
        rows = [*rows, run.prototype_row]
        reports = {**reports, "entry-aware-prototype": run.prototype_report}
    persisted: list[dict[str, object]] = []
    for row in rows:
        strategy = cast(str, row.get("summary_trim_strategy", "head_tail"))
        report_key = (
            "entry-aware-prototype" if strategy != "head_tail" else cast(str, row["cell_id"])
        )
        report = reports[report_key]
        out = {**row, "model_family": run.model_family, "model": run.model, "strategy": strategy}
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=f"{run.model_family}-{row['cell_id']}-{strategy}",
            config={
                "category": METHOD,
                "study_id": design["study_id"],
                "study_kind": design["study_kind"],
                "model_family": run.model_family,
                "model": run.model,
                "backend": run.backend,
                "seed": design["seed"],
                "cell": out,
            },
            metrics={
                "objective_score": report.result.objective_score,
                "reliability": report.reliability,
                "tokens_per_s": (
                    run.prototype_tokens_per_s if strategy != "head_tail" else run.tokens_per_s
                ),
            },
            case_rows=cast(list[dict[str, object]], row["cases"]),
            mirror=mirror,
        )
        out["manifest"] = str(paths["manifest"])
        persisted.append(out)
    return persisted
