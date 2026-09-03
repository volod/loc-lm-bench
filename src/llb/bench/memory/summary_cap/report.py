"""Rendering and persistence for the summarize-input-cap study."""

from pathlib import Path
from typing import cast

from llb.artifacts.runs.members import study_analysis, study_design, table_report
from llb.bench.memory.summary_cap.reading import METHOD, READING_EXACT, STUDY_KIND
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_summary_cap_table(analysis: dict[str, object]) -> str:
    """Render the control recheck, the model-free probe, every arm's ladder, and both readings."""
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
    lines.extend(_probe_lines(cast(list[dict[str, object]], analysis["fold_input_probe"])))
    lines.extend(_cell_lines(cast(list[dict[str, object]], analysis["cells"])))
    for arm in cast(list[dict[str, object]], analysis["arms"]):
        lines.extend(_arm_lines(arm))
    lines.extend(_elision_lines(analysis))
    lines.extend(["", "operator lines"])
    lines.extend(
        f"- {line}"
        for line in cast(list[str], analysis["operator_lines"]) or ["(none: no arm resolved)"]
    )
    lines.extend(
        [
            "",
            f"summarize-input-cap reading: [{analysis['summary_cap_reading']}] {analysis['reason']}",
            f"elision reading: [{analysis['elision_reading']}] {analysis['elision_reason']}",
            f"shipped summarize-input cap: {analysis['shipped_summary_input_cap']}",
        ]
    )
    return "\n".join(lines)


def _probe_lines(probes: list[dict[str, object]]) -> list[str]:
    """The deterministic prediction, which owes nothing to the run that follows it."""
    header = f"{'arm':<16} {'cell':<22} {'guard':>6} {'fold':>4} {'offered':>8} {'elided':>7} folds"
    lines = ["", "model-free summarizer-input probe (oracle play)", header, "-" * len(header)]
    for probe in probes:
        lines.append(
            f"{cast(str, probe['arm_id']):<16} {cast(str, probe['cell_id']):<22} "
            f"{cast(int, probe['max_prompt_chars']):>6d} {cast(int, probe['fold_step']):>4d} "
            f"{cast(int, probe['summary_input_chars']):>8d} "
            f"{cast(int, probe['summary_input_elided_chars']):>7d} "
            f"{cast(int, probe['n_compactions'])}"
        )
    return lines


def _cell_lines(cells: list[dict[str, object]]) -> list[str]:
    if not cells:
        return []
    header = (
        f"{'cell':<26} {'arm':<16} {'guard':>6} {'fold':>4} {'cap tok':>9} {'compact tok':>11} "
        f"{'d(tok)':>10} {'elided':>7} {'compl':>6} {'side':<16} valid"
    )
    lines = ["", "cells", header, "-" * len(header)]
    for cell in cells:
        evidence = cast(dict[str, object], cell["cost_evidence"])
        delta = cast(dict[str, float], evidence["compact_minus_cap_total_model_input_tokens"])
        lines.append(
            f"{cast(str, cell['declared_cell_id']):<26} {cast(str, cell['arm_id']):<16} "
            f"{cast(int, cell['max_prompt_chars']):>6d} "
            f"{cast(int, cell['predicted_fold_step']):>4d} "
            f"{cast(float, cell['cap_mean_total_model_input_tokens']):>9.1f} "
            f"{cast(float, cell['compact_mean_total_model_input_tokens']):>11.1f} "
            f"{delta['mean']:>+10.1f} "
            f"{cast(float, cell['compact_mean_summary_input_elided_chars']):>7.0f} "
            f"{cast(float, cell['compact_completion']):>6.3f} "
            f"{cast(str, cell['measured_side']):<16} {str(cell['valid']).lower()}"
        )
    return lines


def _arm_lines(arm: dict[str, object]) -> list[str]:
    header = (
        f"{'fold step':<9} {'guard interval':<20} {'guards':<16} {'mean d(tok)':>12} "
        f"{'spread':>8} {'ctrl':>6} {'summ':>6} {'band':>8} {'within':>6} side"
    )
    lines = [
        "",
        f"arm {arm['arm_id']} (summary_input_cap={arm['summary_input_cap']}, role={arm['role']})",
        header,
        "-" * len(header),
    ]
    for step in cast(list[dict[str, object]], arm["steps"]):
        lines.append(
            f"{cast(int, step['fold_step']):<9d} "
            f"{_half_open(step['guard_interval']):<20} {str(step['guards']):<16} "
            f"{cast(float, step['mean_cost_delta']):>+12.1f} {cast(float, step['spread']):>8.1f} "
            f"{_optional(step['controller_prompt_spread']):>6} "
            f"{_optional(step['summarizer_prompt_spread']):>6} "
            f"{cast(float, step['equivalence_band']):>8.1f} "
            f"{str(step['within_band']).lower():>6} {step['side']}"
        )
    lines.append(
        f"arm {arm['arm_id']}: [{arm['ladder_reading']}] last compact-cheaper fold step "
        f"{arm['last_compact_cheaper_fold_step']}, within-step residual "
        f"{cast(float, arm['within_step_residual_tokens']):.1f} tok "
        f"(summarizer {_optional(arm['within_step_summarizer_residual_tokens'])}, controller "
        f"{_optional(arm['within_step_controller_residual_tokens'])}), summarizer offered "
        f"{cast(float, arm['mean_summary_input_chars']):.0f} chars of which "
        f"{cast(float, arm['mean_summary_input_elided_chars']):.0f} elided"
    )
    return lines


def _elision_lines(analysis: dict[str, object]) -> list[str]:
    completion = cast(dict[str, object] | None, analysis["elision_completion_delta"])
    if completion is None:
        return []
    delta = cast(dict[str, float], completion["delta"])
    return [
        "",
        f"elision priced against completion: reference arm elided up to "
        f"{cast(float, analysis['reference_elided_chars']):.0f} chars of summarizer input; "
        f"step-aligned minus reference compact completion "
        f"{delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}] "
        f"(w/l/t {completion['wins']}/{completion['losses']}/{completion['ties']}, "
        f"sign-test p={cast(float, completion['sign_test_p']):.4f}) -> "
        f"[{analysis['elision_completion_delta_reading']}]",
    ]


def _half_open(interval: object) -> str:
    low, high = cast(list[int], interval)
    return f"[{low}, {high})"


def _optional(value: object) -> str:
    return "-" if value is None else f"{cast(float, value):.1f}"


def persist_summary_cap(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the aggregate, the immutable design, and the rendered arms."""
    cells = cast(list[dict[str, object]], analysis["cells"])
    arms = cast(list[dict[str, object]], analysis["arms"])
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
            "objective_score": 1.0 if analysis["summary_cap_reading"] == READING_EXACT else 0.0,
            "reliability": (
                sum(bool(arm["ladder_confirms_boundary"]) for arm in arms) / len(arms)
                if arms
                else 0.0
            ),
            "tokens_per_s": tokens_per_s,
        },
        case_rows=cells,
        mirror=mirror,
        artifacts=[
            study_design("summary-input-cap-design.json", design),
            study_analysis("summary-input-cap-analysis.json", analysis),
            table_report("summary-input-cap.md", "Compact summarize-input cap", table),
        ],
    )
