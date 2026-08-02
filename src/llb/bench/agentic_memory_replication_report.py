"""Rendering and persistence for compact-memory second-family replication."""

import json
from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_replication import METHOD, STUDY_KIND
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_replication_table(analysis: dict[str, object]) -> str:
    """Render eligibility, tighter evidence, boundary geometry, and cost."""
    lines = ["control pilots"]
    for row in cast(list[dict[str, object]], analysis["control_pilots"]):
        lines.append(
            f"{row['model']}: completion={cast(float, row['completion']):.3f} "
            f"required={cast(float, row['minimum_completion_rate']):.3f} "
            f"eligible={str(row['eligible']).lower()}"
        )
    rows = cast(list[dict[str, object]], analysis["matrix_rows"])
    if rows:
        header = (
            f"{'cell':<24} {'guard':>6} {'cap':>6} {'compact':>7} {'d(comp)':>8} "
            f"{'cap ovf':>7} {'d(tok)':>9} {'active':>6} verdict"
        )
        lines.extend(["", "replication cells", header, "-" * len(header)])
        for row in rows:
            paired = cast(dict[str, dict[str, object]], row["paired"])
            completion = cast(dict[str, float], paired["completion"]["delta"])
            cost = cast(dict[str, float], paired["total_model_input_tokens"]["delta"])
            lines.append(
                f"{cast(str, row['cell_id']):<24} "
                f"{cast(int, row['max_prompt_chars']):>6d} "
                f"{cast(float, row['cap_completion']):>6.3f} "
                f"{cast(float, row['compact_completion']):>7.3f} "
                f"{completion['mean']:>+8.3f} "
                f"{cast(int, row['cap_context_overflows']):>7d} "
                f"{cost['mean']:>+9.1f} "
                f"{cast(float, row['compaction_activation_rate']):>6.3f} {row['verdict']}"
            )
    lines.extend(
        [
            "",
            f"replication reading: [{analysis['replication_reading']}] {analysis['reason']}",
            f"boundary directional evidence: {analysis['boundary_directional_evidence']}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def persist_replication(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the aggregate and immutable design beside linked source cells."""
    rows = cast(list[dict[str, object]], analysis["matrix_rows"])
    core = [row for row in rows if not row["require_cap_fits"]]
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
                sum(row["verdict"] == "prefer_compact" for row in core) / len(core) if core else 0.0
            ),
            "reliability": 1.0 if analysis["selected_candidate"] is not None else 0.0,
            "tokens_per_s": tokens_per_s,
        },
        case_rows=rows,
        mirror=mirror,
        artifacts={
            "replication-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "replication-analysis.json": json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            "replication-comparison.md": (
                "# Compact-memory transfer replication\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
