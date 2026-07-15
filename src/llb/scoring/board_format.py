"""Focused board format implementation."""

from llb.core.contracts.results import BoardRow
from llb.scoring.aggregate import quality_signals
from llb.scoring.leaderboard import DEFAULT_WEIGHT_JUDGE, ModelResult


def ranking_policy_note(
    results: list[ModelResult], judge_trusted: bool, weight_judge: float = DEFAULT_WEIGHT_JUDGE
) -> str:
    """A human-readable statement of the (policy-dependent) ranking method, so the weighted
    blend is never silently applied: which signals feed the average rank, and the judge weight."""
    signals = quality_signals([r for r in results if r.feasible], judge_trusted) if results else []
    names = [s.replace("_score", "") for s in signals]
    judge = (
        f"judge trusted (blend weight {weight_judge:g})"
        if judge_trusted
        else "judge DEMOTED (objective ranks alone)"
    )
    return f"policy: average rank over [{', '.join(names)}]; {judge}"


def _board_cell(row: BoardRow, key: str) -> str:
    ci = row.get("quality_ci")
    flag = "~" if row["unresolved"] else ""
    mapping = {
        "rank": "-" if row["rank"] is None else str(row["rank"]),
        "model": row["model"],
        "backend": row["backend"],
        "avg_rank": "-" if row["avg_rank"] == float("inf") else f"{row['avg_rank']:.2f}",
        "quality": f"{row['quality']:.3f}{flag}",
        "ci": "-" if ci is None else f"[{ci[0]:.2f},{ci[1]:.2f}]",
        "tok/s": f"{row['tokens_per_s']:.1f}",
        "vram_mb": "-" if row["peak_vram_mb"] is None else f"{row['peak_vram_mb']:.0f}",
        "P": "*" if row["pareto"] else "",
    }
    return mapping[key]


def format_board(rows: list[BoardRow], policy: str | None = None) -> str:
    """ASCII board: rank, avg-rank, quality (with CI + '~' for unresolved), Pareto star.
    Pass `policy` (see `ranking_policy_note`) to print the ranking method above the table."""
    headers = ["rank", "model", "backend", "avg_rank", "quality", "ci", "tok/s", "vram_mb", "P"]

    table = [[_board_cell(row, h) for h in headers] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(headers)
    ]
    out = []
    if policy:
        out.append(policy)
    out += [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    out.append(
        "P = Pareto-optimal (quality/speed/VRAM); ~ = CI overlaps the model above "
        "(rank not statistically resolved)"
    )
    return "\n".join(out)
