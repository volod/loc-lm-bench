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
            f"fit prediction: [{analysis['fit_prediction_reading']}] "
            f"{analysis['fit_prediction_reason']}",
            f"span correction: [{analysis['span_slope_reading']}] {analysis['span_slope_reason']}",
            f"level transfer: [{analysis['level_transfer_reading']}] "
            f"{analysis['level_transfer_reason']}",
            f"ladder fully powered: {str(analysis['ladder_fully_powered']).lower()} -- "
            f"{analysis['ladder_coverage_reason']}",
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
    lines.extend(_guard_fit_lines(family))
    lines.append(f"  powered-fold reason: {family['powered_fold_reason']}")
    lines.append(f"  mechanism: [{family['mechanism_reading']}] {family['mechanism_reason']}")
    return lines


def _guard_fit_lines(family: dict[str, object]) -> list[str]:
    """What each fitted rung cost this family: the guard it moved to and why it moved there."""
    lines: list[str] = []
    for fit in cast(list[dict[str, object]], family.get("guard_fits", [])):
        lines.append(
            f"  guard fit {fit['cell_id']}: declared={fit['declared_max_prompt_chars']} "
            f"fitted={fit['fitted_max_prompt_chars']} "
            f"median-fold-length={fit['median_fold_length_chars']} "
            f"median-step-entry={fit['median_step_entry_chars']} "
            f"(oracle {fit['oracle_step_entry_chars']}) "
            f"predicted={fit['predicted_target_cases']} measured={fit['measured_target_cases']} "
            f"[{fit['fit_reading']}] {fit['fit_reason']}"
        )
        lines.append(
            f"    calibration: step-length [{fit['step_length_reading']}] "
            f"prediction-error={fit['prediction_error_cases']:+d} cases; "
            f"fold length replayed at {fit['median_fold_length_chars']} from "
            f"{fit['fold_length_source']}, this cell measured "
            f"{fit['median_fitted_cell_fold_length_chars']} "
            f"({fit['fold_length_replay_error_chars']:+d})"
        )
        lines.append(
            f"    span: [{fit['fold_span_reading']}] replayed across "
            f"{fit['anchor_fold_span_chars']}-char folds writing "
            f"{fit['anchor_fold_length_chars']} and {fit['second_fold_span_chars']}-char folds "
            f"writing {fit['second_fold_length_chars']} "
            f"({fit['chars_written_per_offered_char']:+.5f} written per offered char); this cell "
            f"folded {fit['fitted_cell_fold_span_range']}-char spans and wrote a median "
            f"{fit['median_fitted_cell_fold_length_chars']}, which the span-aware replay missed "
            f"per fold by {fit['span_replay_error_chars']:+d} against the flat replay's "
            f"{fit['fold_length_replay_error_chars']:+d}"
        )
        lines.append(
            f"    countable: fold-length margin {fit['fold_count_margin_chars']} chars at the "
            f"fitted guard against {fit['declared_fold_count_margin_chars']} at the declared "
            f"{fit['declared_max_prompt_chars']}; replay error inside the margin: "
            f"{str(fit['prediction_within_fold_length_margin']).lower()}"
        )
        lines.append(
            f"    count interval: fitted {_interval(fit['predicted_target_cases_interval'])} "
            f"[{fit['count_reading']}], declared {fit['declared_max_prompt_chars']} "
            f"{_interval(fit['declared_target_cases_interval'])} "
            f"[{fit['declared_count_reading']}]; measured {fit['measured_target_cases']} inside "
            f"the fitted interval: {str(fit['measured_within_predicted_interval']).lower()}; "
            f"{fit['count_reason']}"
        )
        lines.append(
            f"    level: [{fit['level_transfer_reading']}] a case's control fold length against "
            f"its own first fold here correlates {fit['level_transfer_correlation']:+.2f} over "
            f"{fit['level_transfer_pairs']} paired cases; case-to-case spread "
            f"{fit['fold_length_spread_chars']} chars over {fit['fold_length_range_chars']}"
        )
    return lines


def _interval(interval: object) -> str:
    """A count interval as the table prints it, or the refusal when nothing was measured."""
    bounds = cast(list[int], interval or [])
    return f"{bounds[0]}-{bounds[1]} cases" if bounds else "unmeasured"


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
        study_id=cast(str, design["study_id"]),
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
