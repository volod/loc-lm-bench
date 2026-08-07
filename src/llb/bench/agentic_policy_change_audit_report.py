"""Rendering and persistence for the policy-change evidence audit."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_policy_change_audit import (
    VERDICT_CHANGED,
    VERDICT_INVARIANT,
    VERDICT_NOT_APPLICABLE,
    PolicyChange,
)
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths

METHOD = "agentic-policy-change-audit"
# A cell whose arms all sent identical bytes and diverged only on the prompt the guard refused.
# Worth naming: the model saw nothing move, and the episode still ended somewhere else.
REFUSED_PROMPT_NOTE = " (the prompt the guard refused, never sent)"


def policy_change_summary(
    audits: dict[str, list[dict[str, object]]], change: PolicyChange
) -> dict[str, object]:
    """Counts plus the named cells the change invalidates -- the list a re-run is scoped to."""
    rows = [row for study_rows in audits.values() for row in study_rows]
    invalidated = [
        {
            "study_kind": kind,
            "cell_id": row["cell_id"],
            "depth": row["depth"],
            "compact_share": row["compact_share"],
            "max_prompt_chars": row["max_prompt_chars"],
            "changed_arms": row["changed_arms"],
            "refused_prompt_only": row["refused_prompt_only"],
            "candidate_overflows": row["candidate_overflows"],
            "first_divergent_step": row["first_divergent_step"],
        }
        for kind, study_rows in audits.items()
        for row in study_rows
        if row["verdict"] == VERDICT_CHANGED
    ]
    return {
        "policy_fields": list(change.fields),
        "baseline": dict(change.baseline),
        "candidate": dict(change.candidate),
        "change_label": change.label,
        "compound": change.is_compound,
        "n_cells": len(rows),
        "n_prompt_invariant": sum(row["verdict"] == VERDICT_INVARIANT for row in rows),
        # Cells the change cannot describe, because they pin every moved field as their own axis.
        "n_not_applicable": sum(row["verdict"] == VERDICT_NOT_APPLICABLE for row in rows),
        # Cells that pin PART of a compound change: audited on the fields they do not declare.
        "n_partially_applicable": sum(
            bool(row["not_applicable_fields"]) and row["verdict"] != VERDICT_NOT_APPLICABLE
            for row in rows
        ),
        "n_invalidated": len(invalidated),
        # Cells that stop FITTING their published guard under the candidate: re-measuring them
        # needs a new guard, not a re-run at the old one.
        "n_candidate_overflow": sum(bool(row["candidate_overflows"]) for row in rows),
        "invalidated": invalidated,
        "studies_invalidated": sorted({cast(str, row["study_kind"]) for row in invalidated}),
    }


def audited_axis(summary: dict[str, object]) -> str:
    """How a message names the field(s) a skipped cell pins as its own study axis."""
    fields = cast(list[str], summary["policy_fields"])
    return fields[0] if len(fields) == 1 else "an audited constant"


def partial_note(summary: dict[str, object], *, indent: str = "") -> list[str]:
    """The one line a compound change owes cells that declare part of it themselves."""
    partial = cast(int, summary["n_partially_applicable"])
    if not partial:
        return []
    return [
        f"{indent}{partial} cell(s) declare part of this change as their own study axis, so they",
        f"{indent}keep their own value for that field and are audited on the rest of the change.",
    ]


def format_policy_change_table(
    audits: dict[str, list[dict[str, object]]], summary: dict[str, object]
) -> str:
    """Render every audited cell, then the scoped re-run list the change actually implies."""
    lines = [f"policy change: {summary['change_label']}"]
    if summary["compound"]:
        lines.extend(
            [
                f"  {len(cast(list[str], summary['policy_fields']))} constants move together and "
                "are audited as ONE change: the baseline arm replays the whole",
                "  baseline policy and the candidate arm the whole candidate policy, so neither "
                "arm is a configuration",
                "  that never shipped",
            ]
        )
    header = (
        f"{'study':<34} {'cell':<26} {'depth':>5} {'share':>5} {'guard':>6} {'arms changed':<26} "
        f"{'step':>4} verdict"
    )
    lines.extend(["", "replayed cells (oracle play, no GPU)", header, "-" * len(header)])
    for kind, rows in audits.items():
        for row in rows:
            changed = ",".join(cast(list[str], row["changed_arms"])) or "-"
            step = row["first_divergent_step"]
            lines.append(
                f"{kind:<34} {cast(str, row['cell_id']):<26} {cast(int, row['depth']):>5d} "
                f"{cast(float, row['compact_share']):>5.2f} "
                f"{cast(int, row['max_prompt_chars']):>6d} {changed:<26} "
                f"{'-' if step is None else f'{cast(int, step):4d}'} {row['verdict']}"
            )
    lines.extend(["", _verdict_line(summary), *partial_note(summary)])
    if summary["invalidated"]:
        lines.append("")
        lines.append("re-run scope (these published numbers, and no others)")
        lines.extend(format_invalidated_cells(summary))
    return "\n".join(lines)


def format_invalidated_cells(summary: dict[str, object], *, indent: str = "") -> list[str]:
    """One line per invalidated cell: which number to re-measure, and where the change bites.

    Shared by the audit table and the CI pin gate, so an operator reads the same re-run scope
    whether they asked the question or the build asked it for them.
    """
    return [
        f"{indent}- {cell['study_kind']} {cell['cell_id']}: depth {cell['depth']}, guard "
        f"{cell['max_prompt_chars']}, arms {','.join(cast(list[str], cell['changed_arms']))}"
        f", first divergent model call {cell['first_divergent_step']}"
        f"{REFUSED_PROMPT_NOTE if cell['refused_prompt_only'] else ''}"
        f"{_overflow_note(cast(list[str], cell['candidate_overflows']))}"
        for cell in cast(list[dict[str, object]], summary["invalidated"])
    ]


def _overflow_note(arms: list[str]) -> str:
    """Named on the cell that needs it: the candidate does not fit this cell's published guard."""
    if not arms:
        return ""
    return f" -- and {','.join(arms)} no longer fits this guard under the candidate"


def _verdict_line(summary: dict[str, object]) -> str:
    invalidated = cast(int, summary["n_invalidated"])
    skipped = cast(int, summary["n_not_applicable"])
    tail = (
        f" ({skipped} cell(s) pin {audited_axis(summary)} as their own study axis, so the change "
        "does not describe them)"
        if skipped
        else ""
    )
    if not invalidated:
        return (
            f"verdict: every one of the {cast(int, summary['n_cells']) - skipped} applicable "
            "published cells sends bit-identical prompts under both values, so this change "
            f"invalidates NO published number{tail}"
        )
    return (
        f"verdict: {summary['n_prompt_invariant']}/{cast(int, summary['n_cells']) - skipped} "
        f"applicable published cells are prompt-invariant; {invalidated} must be re-measured "
        f"across {len(cast(list[str], summary['studies_invalidated']))} study/studies{tail}"
    )


def persist_policy_change_audit(
    audits: dict[str, list[dict[str, object]]],
    summary: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the audit so a policy change carries the record of what it invalidated."""
    rows = [
        {"study_kind": kind, **row} for kind, study_rows in audits.items() for row in study_rows
    ]
    return persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name=f"policy-change-{'-'.join(cast(list[str], summary['policy_fields']))}",
        config={"category": METHOD, "summary": summary},
        metrics={
            "objective_score": 1.0 if not summary["n_invalidated"] else 0.0,
            "reliability": (
                cast(int, summary["n_prompt_invariant"]) / cast(int, summary["n_cells"])
                if summary["n_cells"]
                else 0.0
            ),
            # No model runs; the audit is pure replay over the deterministic tool world.
            "tokens_per_s": 0.0,
        },
        case_rows=rows,
        mirror=mirror,
        artifacts={
            "policy-change-audit.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
            "policy-change-audit.md": (
                "# Agent policy-change evidence audit\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
