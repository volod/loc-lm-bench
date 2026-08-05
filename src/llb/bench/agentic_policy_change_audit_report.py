"""Rendering and persistence for the policy-change evidence audit."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_policy_change_audit import (
    VERDICT_CHANGED,
    VERDICT_INVARIANT,
    VERDICT_NOT_APPLICABLE,
)
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths

METHOD = "agentic-policy-change-audit"


def policy_change_summary(
    audits: dict[str, list[dict[str, object]]], *, field: str, baseline: object, candidate: object
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
            "first_divergent_step": row["first_divergent_step"],
        }
        for kind, study_rows in audits.items()
        for row in study_rows
        if row["verdict"] == VERDICT_CHANGED
    ]
    return {
        "policy_field": field,
        "baseline": baseline,
        "candidate": candidate,
        "n_cells": len(rows),
        "n_prompt_invariant": sum(row["verdict"] == VERDICT_INVARIANT for row in rows),
        # Cells the change cannot describe, because they pin the field as their own study axis.
        "n_not_applicable": sum(row["verdict"] == VERDICT_NOT_APPLICABLE for row in rows),
        "n_invalidated": len(invalidated),
        "invalidated": invalidated,
        "studies_invalidated": sorted({cast(str, row["study_kind"]) for row in invalidated}),
    }


def format_policy_change_table(
    audits: dict[str, list[dict[str, object]]], summary: dict[str, object]
) -> str:
    """Render every audited cell, then the scoped re-run list the change actually implies."""
    lines = [
        f"policy change: {summary['policy_field']} "
        f"{summary['baseline']!r} -> {summary['candidate']!r}",
    ]
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
    lines.extend(["", _verdict_line(summary)])
    if summary["invalidated"]:
        lines.append("")
        lines.append("re-run scope (these published numbers, and no others)")
        for cell in cast(list[dict[str, object]], summary["invalidated"]):
            lines.append(
                f"- {cell['study_kind']} {cell['cell_id']}: depth {cell['depth']}, guard "
                f"{cell['max_prompt_chars']}, arms {','.join(cast(list[str], cell['changed_arms']))}"
                f", first divergent model call {cell['first_divergent_step']}"
            )
    return "\n".join(lines)


def _verdict_line(summary: dict[str, object]) -> str:
    invalidated = cast(int, summary["n_invalidated"])
    skipped = cast(int, summary["n_not_applicable"])
    tail = (
        f" ({skipped} cell(s) pin {summary['policy_field']} as their own study axis, so the change "
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
        run_name=f"policy-change-{summary['policy_field']}",
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
