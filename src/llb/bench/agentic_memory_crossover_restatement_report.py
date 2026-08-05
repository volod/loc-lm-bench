"""Rendering and persistence for the published-crossover restatement."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_crossover_restatement_reading import (
    METHOD,
    READING_ALL_INVARIANT,
    READING_UNCHANGED,
    STUDY_KIND,
)
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_restatement_table(analysis: dict[str, object]) -> str:
    """Render the audit, whatever had to be re-measured, and every restated crossover."""
    lines = ["control recheck"]
    control = cast(dict[str, object] | None, analysis["control_recheck"])
    if control is None:
        lines.append("(not run)")
    else:
        lines.append(
            f"{control['model']}: completion={cast(float, control['completion']):.3f} "
            f"required={cast(float, control['minimum_completion_rate']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
    lines.extend(_audit_lines(cast(dict[str, object], analysis["audit"])))
    lines.extend(_restated_lines(cast(list[dict[str, object]], analysis["restated_cells"])))
    lines.extend(_surface_lines(cast(list[dict[str, object]], analysis["restated_depth_surface"])))
    lines.extend(_crossover_lines(cast(list[dict[str, object]], analysis["crossovers"])))
    lines.extend(["", "operator lines"])
    lines.extend(f"- {line}" for line in cast(list[str], analysis["operator_lines"]))
    lines.extend(
        [
            "",
            f"restatement reading: [{analysis['restatement_reading']}] {analysis['reason']}",
            f"shipped summarize-input cap: {analysis['shipped_summary_input_cap']}",
        ]
    )
    return "\n".join(lines)


def _audit_lines(audit: dict[str, object]) -> list[str]:
    summary = cast(dict[str, object], audit["summary"])
    header = (
        f"{'study':<34} {'cell':<26} {'depth':>5} {'share':>5} {'guard':>6} {'offered':>8} "
        f"{'elided':>7} verdict"
    )
    lines = ["", "model-free bound audit (oracle play, no GPU)", header, "-" * len(header)]
    for kind, rows in cast(dict[str, list[dict[str, object]]], audit["per_study"]).items():
        for row in rows:
            lines.append(
                f"{kind:<34} {cast(str, row['cell_id']):<26} {cast(int, row['depth']):>5d} "
                f"{cast(float, row['compact_share']):>5.2f} "
                f"{cast(int, row['max_prompt_chars']):>6d} "
                f"{cast(int, row['summary_input_chars']):>8d} "
                f"{cast(int, row['trigger_elided_chars']):>7d} {row['verdict']}"
            )
    lines.append(
        f"audit: {summary['n_bound_invariant']}/{summary['n_cells']} published cells are "
        f"bit-identical under both bounds; {summary['n_bound_sensitive']} need re-measurement"
    )
    return lines


def _restated_lines(cells: list[dict[str, object]]) -> list[str]:
    if not cells:
        return ["", "re-measured cells: (none needed a run)"]
    header = (
        f"{'cell':<26} {'guard':>6} {'cap tok':>9} {'compact tok':>11} {'d(tok)':>10} "
        f"{'elided':>7} {'side':<16} valid"
    )
    lines = ["", "re-measured under the shipped cap", header, "-" * len(header)]
    for cell in cells:
        evidence = cast(dict[str, object], cell["cost_evidence"])
        delta = cast(dict[str, float], evidence["compact_minus_cap_total_model_input_tokens"])
        lines.append(
            f"{cast(str, cell['cell_id']):<26} {cast(int, cell['max_prompt_chars']):>6d} "
            f"{cast(float, cell['cap_mean_total_model_input_tokens']):>9.1f} "
            f"{cast(float, cell['compact_mean_total_model_input_tokens']):>11.1f} "
            f"{delta['mean']:>+10.1f} "
            f"{cast(float, cell['compact_mean_summary_input_elided_chars']):>7.0f} "
            f"{cast(str, cell['measured_side']):<16} {str(cell['valid']).lower()}"
        )
    return lines


def _surface_lines(surfaces: list[dict[str, object]]) -> list[str]:
    if not surfaces:
        return []
    header = f"{'depth':>5} {'guards':<24} {'bracket':<16} {'crossover':>10} {'ratio':>6} reading"
    lines = [
        "",
        "surface interpolation re-run over the substituted cells",
        header,
        "-" * len(header),
    ]
    for row in surfaces:
        crossover = cast(float | None, row["crossover_max_prompt_chars"])
        ratio = cast(float | None, row["crossover_guard_ratio"])
        lines.append(
            f"{cast(int, row['depth']):>5d} {str(row['guards']):<24} "
            f"{str(row['bracket']):<16} "
            f"{'-' if crossover is None else f'{crossover:10.0f}'} "
            f"{'-' if ratio is None else f'{ratio:6.2f}'} {row['reading']}"
        )
    return lines


def _crossover_lines(crossovers: list[dict[str, object]]) -> list[str]:
    header = (
        f"{'study':<34} {'depth':>5} {'form':<22} {'published':>10} {'restated':>10} "
        f"{'fold':>4} {'same':>5} basis"
    )
    lines = ["", "published crossovers", header, "-" * len(header)]
    for row in crossovers:
        published = cast(float | None, row["published_value"])
        restated = cast(float | None, row["restated_value"])
        lines.append(
            f"{cast(str, row['study_kind']):<34} {cast(int, row['depth']):>5d} "
            f"{cast(str, row['form']):<22} "
            f"{'-' if published is None else f'{published:10.0f}'} "
            f"{'-' if restated is None else f'{restated:10.0f}'} "
            f"{cast(int, row['restated_fold_step']):>4d} "
            f"{str(row['names_same_fold_step']).lower():>5} {row['basis']}"
        )
    return lines


def persist_restatement(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the aggregate, the immutable design, and the rendered restatement."""
    crossovers = cast(list[dict[str, object]], analysis["crossovers"])
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=cast(str, design["study_id"]),
        config={
            "category": METHOD,
            "study_kind": STUDY_KIND,
            "design": design,
            "analysis": analysis,
        },
        metrics={
            "objective_score": (
                1.0
                if analysis["restatement_reading"] in (READING_UNCHANGED, READING_ALL_INVARIANT)
                else 0.0
            ),
            "reliability": (
                sum(bool(row["names_same_fold_step"]) for row in crossovers) / len(crossovers)
                if crossovers
                else 0.0
            ),
            "tokens_per_s": tokens_per_s,
        },
        case_rows=crossovers,
        mirror=mirror,
        artifacts={
            "crossover-restatement-design.json": json.dumps(design, indent=2, sort_keys=True)
            + "\n",
            "crossover-restatement-analysis.json": json.dumps(analysis, indent=2, sort_keys=True)
            + "\n",
            "crossover-restatement.md": (
                "# Published crossovers under the shipped summarize-input cap\n\n```text\n"
                + table
                + "\n```\n"
            ),
        },
    )
