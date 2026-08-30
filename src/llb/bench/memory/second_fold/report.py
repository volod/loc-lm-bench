"""Rendering and persistence for the second-fold trigger restatement."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic.design_fields import as_mapping, as_str
from llb.bench.common import Mirror, persist_category_run
from llb.bench.context_policy.compact_vs_cap_report import METHOD
from llb.bench.context_policy.report import PolicyReport
from llb.bench.memory.second_fold.reading import (
    KIND_EQUAL_TRIGGER,
    READING_COLLAPSES,
    STUDY_KIND,
)
from llb.core.contracts.runs import RunPaths


def format_second_fold_table(analysis: dict[str, object]) -> str:
    """Render the control, every cell's cost and fold regime, the families, and the repeat pair."""
    lines = _control_lines(analysis)
    lines.extend(_cell_lines(analysis))
    lines.extend(_fold_input_lines(analysis))
    lines.extend(_family_lines(analysis))
    lines.extend(_repeat_lines(analysis))
    lines.extend(
        [
            "",
            f"second-fold reading: [{analysis['second_fold_reading']}] {analysis['reason']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def _control_lines(analysis: dict[str, object]) -> list[str]:
    control = cast(dict[str, object] | None, analysis["control_recheck"])
    if control is None:
        return ["control recheck", "(not run)"]
    return [
        "control recheck",
        f"{control['model']}: completion={cast(float, control['completion']):.3f} "
        f"required={cast(float, control['minimum_completion_rate']):.3f} "
        f"eligible={str(control['eligible']).lower()}",
    ]


def _cell_lines(analysis: dict[str, object]) -> list[str]:
    cells = cast(list[dict[str, object]], analysis["cells"])
    if not cells:
        return []
    header = (
        f"{'cell':<28} {'share':>5} {'guard':>6} {'trigger':>7} {'step':>4} {'oracle':>6} "
        f"{'measured folds':<16} {'input tok':>10} {'oracle chars':>12} {'complete':>8} valid"
    )
    lines = ["", "cells", header, "-" * len(header)]
    for cell in cells:
        folds = ",".join(str(value) for value in cast(list[int], cell["measured_fold_counts"]))
        lines.append(
            f"{cast(str, cell['cell_id']):<28} "
            f"{cast(float, cell['compact_share']):>5.2f} "
            f"{cast(int, cell['max_prompt_chars']):>6d} "
            f"{cast(int, cell['compaction_trigger_chars']):>7d} "
            f"{cast(int, cell['first_fold_step']):>4d} "
            f"{cast(int, cell['oracle_folds']):>6d} {folds:<16} "
            f"{cast(float, cell['mean_total_model_input_tokens']):>10.1f} "
            f"{cast(int, cell['oracle_model_input_chars']):>12d} "
            f"{cast(float, cell['completion']):>8.3f} {str(cell['valid']).lower()}"
        )
    return lines


def _fold_input_lines(analysis: dict[str, object]) -> list[str]:
    """What each fold offered the summarizer -- the other way the guard can re-enter the cost."""
    cells = cast(list[dict[str, object]], analysis["cells"])
    if not cells:
        return []
    header = f"{'cell':<28} {'fold':>4} {'episodes':>8} {'offered chars':>13} {'oracle':>8}"
    lines = ["", "per-fold summarize input", header, "-" * len(header)]
    for cell in cells:
        oracle = cast(list[int], cell["oracle_fold_input_chars"])
        for row in cast(list[dict[str, object]], cell["measured_fold_input_chars"]):
            index = int(cast(int, row["fold"])) - 1
            predicted = oracle[index] if index < len(oracle) else 0
            lines.append(
                f"{cast(str, cell['cell_id']):<28} {index + 1:>4d} "
                f"{cast(int, row['n_episodes']):>8d} "
                f"{cast(float, row['mean_offered_chars']):>13.1f} {predicted:>8d}"
            )
    return lines


def _family_lines(analysis: dict[str, object]) -> list[str]:
    families = cast(list[dict[str, object]], analysis["families"])
    if not families:
        return []
    header = (
        f"{'family':<20} {'kind':<14} {'triggers':<16} {'fold steps':<12} {'spread':>10} "
        f"{'band':>9} {'within':>6} separated"
    )
    lines = ["", "families", header, "-" * len(header)]
    for row in families:
        lines.append(
            f"{cast(str, row['family_id']):<20} {cast(str, row['kind']):<14} "
            f"{str(row['triggers']):<16} {str(row['first_fold_steps']):<12} "
            f"{cast(float, row['spread']):>10.1f} "
            f"{cast(float, row['equivalence_band']):>9.1f} "
            f"{str(row['within_band']).lower():>6} {row['separated_members']}"
        )
        for member in cast(list[dict[str, object]], row["member_deltas"]):
            delta = cast(dict[str, float], member["paired_delta"])
            lines.append(
                f"  vs anchor {cast(str, member['cell_id']):<26} "
                f"d={delta['mean']:>+10.1f} [{delta['lo']:>+9.1f}, {delta['hi']:>+9.1f}] "
                f"p={cast(float, member['two_sided_sign_test_p']):.6f} "
                f"separates={str(member['separates']).lower()} "
                f"(expected {str(member['expected_separation']).lower()})"
            )
    return lines


def _repeat_lines(analysis: dict[str, object]) -> list[str]:
    repeat = cast(dict[str, object] | None, analysis["repeat_geometry"])
    if repeat is None:
        return []
    delta = cast(dict[str, float], repeat["paired_delta"])
    return [
        "",
        "repeat geometry (noise floor)",
        f"{repeat['cell_id']} vs {repeat['anchor_cell_id']}: d={delta['mean']:+.1f} tok "
        f"band={cast(float, repeat['equivalence_band']):.1f} "
        f"reproduces={str(repeat['reproduces']).lower()}",
    ]


def persist_second_fold(
    design: dict[str, object],
    analysis: dict[str, object],
    reports: dict[str, PolicyReport],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist one bundle per compact cell, then the aggregate over the same method root."""
    cells = cast(list[dict[str, object]], analysis["cells"])
    for row in cells:
        cell_id = cast(str, row["cell_id"])
        report = reports[cell_id]
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=f"second-fold-{cell_id}",
            config={
                "category": METHOD,
                "study_id": design["study_id"],
                "study_kind": STUDY_KIND,
                "model": as_str(as_mapping(analysis, "held_fixed"), "model"),
                "cell": row,
            },
            metrics={
                "objective_score": report.result.objective_score,
                "reliability": report.reliability,
                "tokens_per_s": tokens_per_s,
            },
            case_rows=cast(list[dict[str, object]], report.rows),
            mirror=mirror,
        )
        row["manifest"] = str(paths["manifest"])
    held = [
        row
        for row in cast(list[dict[str, object]], analysis["families"])
        if row["kind"] == KIND_EQUAL_TRIGGER
    ]
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
                1.0 if analysis["second_fold_reading"] == READING_COLLAPSES else 0.0
            ),
            "reliability": (
                sum(bool(row["within_band"]) for row in held) / len(held) if held else 0.0
            ),
            "tokens_per_s": tokens_per_s,
        },
        case_rows=cells,
        mirror=mirror,
        artifacts={
            "second-fold-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "second-fold-analysis.json": json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            "second-fold.md": (
                "# Compact trigger rule through a second fold\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
