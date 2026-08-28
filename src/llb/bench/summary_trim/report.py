"""Rendering and persistence for the entry-aware summary-fold adoption study."""

import json
from pathlib import Path
from typing import cast

from llb.bench.memory.window_elision.tasks import STRATA
from llb.bench.summary_trim.run import FamilyRun
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths

METHOD = "agentic-summary-trim-adoption"


def format_summary_trim_table(analysis: dict[str, object]) -> str:
    """Render the declared geometry, the per-workload deltas, and the adoption verdict."""
    return "\n".join(
        [
            *_geometry_lines(analysis),
            "",
            *_workload_lines(analysis),
            "",
            *_stratum_lines(analysis),
            *_audit_lines(analysis),
            "",
            f"adoption: [{analysis['adoption_reading']}] {analysis['adoption_reason']}",
            "shipped default changed: false",
        ]
    )


def _geometry_lines(analysis: dict[str, object]) -> list[str]:
    header = (
        f"{'workload':<17} {'folds':>5} {'offered':>8} {'elided':>7} "
        f"{'summary chars head_tail':>24} {'per_entry_head':>15}"
    )
    lines = ["declared geometry (oracle play, no model)", header, "-" * len(header)]
    for row in cast(list[dict[str, object]], analysis["declared_geometry"]):
        head_tail = cast(dict[str, object], row["head_tail"])
        entry = cast(dict[str, object], row["per_entry_head"])
        lines.append(
            f"{cast(str, row['workload']):<17} "
            f"{cast(int, head_tail['n_compactions']):>5d} "
            f"{cast(int, head_tail['summary_input_chars']):>8d} "
            f"{cast(int, head_tail['summary_input_elided_chars']):>7d} "
            f"{cast(int, head_tail['compaction_prompt_chars']):>24d} "
            f"{cast(int, entry['compaction_prompt_chars']):>15d}"
        )
    return lines


def _workload_lines(analysis: dict[str, object]) -> list[str]:
    header = (
        f"{'family':<10} {'workload':<17} {'pairs':>5} {'skip':>4} {'ea wins':>7} {'ht wins':>7} "
        f"{'d(input chars)':>14} {'d(summary chars)':>16} {'d(folds)':>8} reading"
    )
    lines = ["measured deltas (entry_aware minus head_tail)", header, "-" * len(header)]
    for family in cast(list[dict[str, object]], analysis["families"]):
        for row in cast(list[dict[str, object]], family["workloads"]):
            lines.append(
                f"{cast(str, family['model_family']):<10} "
                f"{cast(str, row['workload']):<17} "
                f"{cast(int, row['n_pairs']):>5d} "
                f"{cast(int, row['n_unpaired']):>4d} "
                f"{cast(int, row['entry_aware_wins']):>7d} "
                f"{cast(int, row['head_tail_wins']):>7d} "
                f"{cast(int, row['d_model_input_prompt_chars']):>+14d} "
                f"{cast(int, row['d_summary_prompt_chars']):>+16d} "
                f"{cast(int, row['d_measured_folds']):>+8d} {row['reading']}"
            )
        lines.append(
            f"{'':<10} eligible={str(family['eligible']).lower()} "
            f"({family['eligibility_reason']}) throughput={family['tokens_per_s']:.2f} tok/s"
        )
    return lines


def _stratum_lines(analysis: dict[str, object]) -> list[str]:
    lines = ["middle-critical recovery (evidence strata)"]
    for family in cast(list[dict[str, object]], analysis["families"]):
        strata = cast(dict[str, dict[str, int]], family.get("strata", {}))
        for stratum in STRATA:
            row = strata.get(stratum)
            if row is None:
                continue
            skipped = int(row["n_declared"]) - int(row["n_pairs"])
            note = f"; {skipped} case(s) never folded in one arm" if skipped else ""
            lines.append(
                f"- {family['model_family']} {stratum}: head_tail "
                f"{row['head_tail_completed']}/{row['n_pairs']} -> entry_aware "
                f"{row['entry_aware_completed']}/{row['n_pairs']} usable pair(s) of "
                f"{row['n_declared']} declared "
                f"(ea wins {row['entry_aware_wins']}, ht wins {row['head_tail_wins']}{note})"
            )
    return lines


def _audit_lines(analysis: dict[str, object]) -> list[str]:
    audit = cast(dict[str, object], analysis["policy_change_audit"])
    lines = [
        "",
        f"policy-change audit ({audit['change']}, pinned policy): "
        f"{audit['n_prompt_invariant']}/{audit['n_cells']} published cells prompt-invariant, "
        f"{audit['n_invalidated']} invalidated",
    ]
    lines.extend(f"- retires {cell}" for cell in cast(list[str], audit["invalidated_cells"]))
    lines.extend(
        f"- published value affected: {row}"
        for row in cast(list[str], audit["affected_published_values"])
    )
    return lines


def persist_summary_trim_adoption(
    design: dict[str, object],
    runs: list[FamilyRun],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist every family/workload/arm cell and one cross-family aggregate."""
    persisted: list[dict[str, object]] = []
    for run in runs:
        persisted.extend(_persist_family_cells(run, design, data_dir=data_dir, mirror=mirror))
    objective = (
        sum(float(cast(float, row["completion"])) for row in persisted) / len(persisted)
        if persisted
        else 0.0
    )
    throughputs = [run.tokens_per_s for run in runs if run.tokens_per_s > 0.0]
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
            "tokens_per_s": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        },
        case_rows=persisted,
        mirror=mirror,
        artifacts={
            "summary-trim-adoption-design.json": json.dumps(design, indent=2, sort_keys=True)
            + "\n",
            "summary-trim-adoption-analysis.json": json.dumps(analysis, indent=2, sort_keys=True)
            + "\n",
            "summary-trim-adoption.md": "# Entry-aware summary-fold adoption\n\n```text\n"
            + table
            + "\n```\n",
        },
    )


def _persist_family_cells(
    run: FamilyRun,
    design: dict[str, object],
    *,
    data_dir: Path | str,
    mirror: Mirror | None,
) -> list[dict[str, object]]:
    persisted: list[dict[str, object]] = []
    for row in run.rows:
        key = (cast(str, row["workload"]), cast(str, row["arm"]))
        report = run.reports[key]
        out = {**row, "model_family": run.model_family, "model": run.model}
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=f"{run.model_family}-{key[0]}-{key[1]}",
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
                "tokens_per_s": run.tokens_per_s,
            },
            case_rows=cast(list[dict[str, object]], row["cases"]),
            mirror=mirror,
        )
        out["manifest"] = str(paths["manifest"])
        persisted.append(out)
    return persisted
