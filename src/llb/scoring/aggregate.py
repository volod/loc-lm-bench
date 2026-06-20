"""Rank model results into a leaderboard row (pure Python).

Ranking axis (design): GENERATION quality -- objective reference correctness, blended
with the gated judge ONLY when the judge is trusted, else objective alone. Fit/fail is a
hard filter (infeasible models are listed without a rank). Ties break by tokens/sec
(desc) then peak VRAM (asc), matching the Pareto tie-breaker order.

Milestone 1 produces a single row; the function is written for N so the multi-model
sweep in later milestones reuses it unchanged.
"""

from dataclasses import dataclass

DEFAULT_WEIGHT_JUDGE = 0.5


@dataclass
class ModelResult:
    """One model's scored outcome over the eval set."""

    model: str
    backend: str
    objective_score: float          # mean reference correctness over scored cases
    n_cases: int
    reliability: float = 1.0        # fraction of cases that ended status=ok
    tokens_per_s: float = 0.0
    peak_vram_mb: float | None = None
    judge_score: float | None = None
    feasible: bool = True


def headline_quality(result: ModelResult, judge_trusted: bool,
                     weight_judge: float = DEFAULT_WEIGHT_JUDGE) -> float:
    """Blend objective + judge when trusted; objective alone otherwise."""
    if judge_trusted and result.judge_score is not None:
        return (1.0 - weight_judge) * result.objective_score + weight_judge * result.judge_score
    return result.objective_score


def _vram_key(result: ModelResult) -> float:
    return result.peak_vram_mb if result.peak_vram_mb is not None else float("inf")


def rank_results(results: list[ModelResult], judge_trusted: bool = False,
                 weight_judge: float = DEFAULT_WEIGHT_JUDGE) -> list[dict]:
    """Return ranked row dicts. Feasible models ranked by quality; infeasible appended."""
    feasible = [r for r in results if r.feasible]
    infeasible = [r for r in results if not r.feasible]
    ordered = sorted(
        feasible,
        key=lambda r: (-headline_quality(r, judge_trusted, weight_judge),
                       -r.tokens_per_s, _vram_key(r)),
    )
    rows: list[dict] = []
    for rank, r in enumerate(ordered, 1):
        rows.append(_row(r, rank, judge_trusted, weight_judge))
    for r in infeasible:
        rows.append(_row(r, None, judge_trusted, weight_judge))
    return rows


def _row(r: ModelResult, rank: int | None, judge_trusted: bool, weight_judge: float) -> dict:
    return {
        "rank": rank,
        "model": r.model,
        "backend": r.backend,
        "quality": round(headline_quality(r, judge_trusted, weight_judge), 4),
        "objective": round(r.objective_score, 4),
        "judge": None if r.judge_score is None else round(r.judge_score, 4),
        "reliability": round(r.reliability, 4),
        "tokens_per_s": round(r.tokens_per_s, 2),
        "peak_vram_mb": r.peak_vram_mb,
        "feasible": r.feasible,
        "n_cases": r.n_cases,
    }


def format_table(rows: list[dict]) -> str:
    """Render ranked rows as an ASCII table (judge column omitted when always demoted)."""
    show_judge = any(row.get("judge") is not None for row in rows)
    headers = ["rank", "model", "backend", "quality", "objective"]
    if show_judge:
        headers.append("judge")
    headers += ["reliab", "tok/s", "vram_mb", "feasible"]

    def cell(row: dict, key: str) -> str:
        mapping = {
            "rank": "-" if row["rank"] is None else str(row["rank"]),
            "model": row["model"],
            "backend": row["backend"],
            "quality": f"{row['quality']:.3f}",
            "objective": f"{row['objective']:.3f}",
            "judge": "-" if row.get("judge") is None else f"{row['judge']:.3f}",
            "reliab": f"{row['reliability']:.3f}",
            "tok/s": f"{row['tokens_per_s']:.1f}",
            "vram_mb": "-" if row["peak_vram_mb"] is None else f"{row['peak_vram_mb']:.0f}",
            "feasible": "yes" if row["feasible"] else "NO",
        }
        return mapping[key]

    table = [[cell(row, h) for h in headers] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in table)) if table else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * widths[i] for i in range(len(headers)))]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)
