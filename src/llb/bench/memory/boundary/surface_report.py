"""Rendering and persistence for the cap-fitting boundary surface."""

import json
from pathlib import Path
from typing import cast

from llb.bench.memory.boundary.crossover import READING_BRACKETED
from llb.bench.memory.boundary.surface import METHOD, STUDY_KIND
from llb.bench.common import Mirror, persist_category_run
from llb.core.contracts.runs import RunPaths


def format_surface_table(analysis: dict[str, object]) -> str:
    """Render the control recheck, every cell's cost side, and the interpolated surface."""
    control = cast(dict[str, object] | None, analysis["control_recheck"])
    lines = ["control recheck"]
    if control is None:
        lines.append("(not run)")
    else:
        lines.append(
            f"{control['model']}: completion={cast(float, control['completion']):.3f} "
            f"required={cast(float, control['minimum_completion_rate']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
    cells = cast(list[dict[str, object]], analysis["cells"])
    if cells:
        header = (
            f"{'cell':<22} {'depth':>5} {'guard':>6} {'cap':>6} {'compact':>7} "
            f"{'d(tok)':>10} {'p(cost)':>8} {'active':>6} {'side':<16} {'expected':<16} valid"
        )
        lines.extend(["", "surface cells", header, "-" * len(header)])
        for cell in cells:
            evidence = cast(dict[str, object], cell["cost_evidence"])
            delta = cast(dict[str, float], evidence["compact_minus_cap_total_model_input_tokens"])
            lines.append(
                f"{cast(str, cell['cell_id']):<22} "
                f"{cast(int, cell['depth']):>5d} "
                f"{cast(int, cell['max_prompt_chars']):>6d} "
                f"{cast(float, cell['cap_completion']):>6.3f} "
                f"{cast(float, cell['compact_completion']):>7.3f} "
                f"{delta['mean']:>+10.1f} "
                f"{cast(float, evidence['cost_two_sided_sign_test_p']):>8.4f} "
                f"{cast(float, cell['compaction_activation_rate']):>6.3f} "
                f"{cast(str, cell['measured_side']):<16} "
                f"{cast(str, cell['expected_side']):<16} {str(cell['valid']).lower()}"
            )
    lines.extend(_margin_lines(analysis, cells))
    rows = cast(list[dict[str, object]], analysis["depth_surface"])
    if rows:
        lines.extend(["", "depth surface"])
        for row in rows:
            crossover = row["crossover_max_prompt_chars"]
            rendered = (
                f"{cast(float, crossover):.0f} chars "
                f"({cast(float, cast(float, row['crossover_guard_ratio'])):.2f}x peak) "
                f"bracket={row['bracket']}"
                if row["reading"] == READING_BRACKETED
                else "none"
            )
            lines.append(
                f"depth {cast(int, row['depth']):>2d}: cap peak="
                f"{cast(int, row['cap_peak_prompt_chars'])} chars guards={row['guards']} "
                f"crossover={rendered} [{row['reading']}]"
            )
    lines.extend(["", "routing rule"])
    lines.extend(f"- {line}" for line in cast(list[str], analysis["routing_rule"]) or ["(none)"])
    lines.extend(
        [
            "",
            f"surface reading: [{analysis['surface_reading']}] {analysis['reason']}",
            f"cells matching their predeclared side: {analysis['expectation_matches']}"
            f"/{len(cells)}",
            "shipped default changed: false",
        ]
    )
    return "\n".join(lines)


def _margin_lines(analysis: dict[str, object], cells: list[dict[str, object]]) -> list[str]:
    """The imperfect-play safety margin per depth, and what the run's own episodes spent."""
    margins = cast(dict[str, dict[str, object]], analysis.get("cap_peak_margin", {}))
    if not margins:
        return []
    lines = ["", "imperfect-play safety margin"]
    for depth in sorted(margins, key=int):
        margin = margins[depth]
        lines.append(
            f"depth {int(depth):>2d}: perfect play={cast(int, margin['perfect_play_peak_chars'])} "
            f"worst case={cast(int, margin['worst_case_peak_chars'])} chars "
            f"(+{cast(int, margin['margin_chars'])}, "
            f"{cast(float, margin['margin_ratio']):.3f}x) over "
            f"{cast(int, margin['budgeted_extra_steps'])} budgeted extra step(s); "
            f"observed={_observed_extra(cells, int(depth))}"
        )
    return lines


def _observed_extra(cells: list[dict[str, object]], depth: int) -> str:
    """The largest extra step count any readable bundle at this depth recorded."""
    values = [
        cast(int, arm["max_extra_steps"])
        for cell in cells
        if int(cast(int, cell["depth"])) == depth
        for arm in cast(dict[str, dict[str, object]], cell.get("observed_extra_steps", {})).values()
        if arm["read"]
    ]
    return f"{max(values)} extra step(s)" if values else "(bundles not on this host)"


def persist_surface(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    table: str,
    tokens_per_s: float,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the aggregate, the immutable design, and the rendered surface."""
    rows = cast(list[dict[str, object]], analysis["cells"])
    depth_rows = cast(list[dict[str, object]], analysis["depth_surface"])
    bracketed = sum(row["reading"] == READING_BRACKETED for row in depth_rows)
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
            "objective_score": bracketed / len(depth_rows) if depth_rows else 0.0,
            "reliability": 1.0 if rows and all(row["valid"] for row in rows) else 0.0,
            "tokens_per_s": tokens_per_s,
        },
        case_rows=rows,
        mirror=mirror,
        study_id=cast(str, design["study_id"]),
        artifacts={
            "boundary-surface-design.json": json.dumps(design, indent=2, sort_keys=True) + "\n",
            "boundary-surface-analysis.json": json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            "boundary-surface.md": (
                "# Compact-memory cap-fitting boundary surface\n\n```text\n" + table + "\n```\n"
            ),
        },
    )
