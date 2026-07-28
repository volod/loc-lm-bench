"""Focused leaderboard implementation.

RAG rows may provide a declared fact/format ``ranking_score``; other tiers and legacy rows rank on
their objective. A trusted judge blends with that tier-specific base quality.
"""

import random
from dataclasses import dataclass, field
from llb.core.contracts.results import LeaderboardRow
from llb.scoring.verbosity import policy_description

DEFAULT_WEIGHT_JUDGE = 0.5

TIER_PRIVATE = "private"  # Tier-2 private gold-set metrics


@dataclass
class ModelResult:
    """One model's scored outcome over the eval set."""

    model: str
    backend: str
    objective_score: float  # mean reference correctness over scored cases
    n_cases: int
    reliability: float = 1.0  # fraction of cases that ended status=ok
    tokens_per_s: float = 0.0
    peak_vram_mb: float | None = None
    judge_score: float | None = None
    semantic_score: float | None = None
    # RAG-only declared policy. Other benchmark tiers and legacy bundles leave this absent and
    # continue to rank on objective_score.
    ranking_score: float | None = None
    token_precision: float | None = None
    token_recall: float | None = None
    found_rate: float | None = None
    mean_completion_tokens: float | None = None
    feasible: bool = True
    tier: str = TIER_PRIVATE
    # Per-case score series -> bootstrap CIs (optional; aligned by case order).
    case_objectives: list[float] = field(default_factory=list)
    case_ranking: list[float] = field(default_factory=list)
    case_semantic: list[float] = field(default_factory=list)
    case_judge: list[float] = field(default_factory=list)


def per_case_quality(
    result: ModelResult, judge_trusted: bool, weight_judge: float = DEFAULT_WEIGHT_JUDGE
) -> list[float]:
    """The per-case headline series: declared RAG ranking scores when present, else objectives."""
    base = result.case_ranking or result.case_objectives
    if judge_trusted and result.case_judge and len(result.case_judge) == len(base):
        return [
            (1.0 - weight_judge) * obj + weight_judge * jud
            for obj, jud in zip(base, result.case_judge)
        ]
    return list(base)


def base_quality(result: ModelResult) -> float:
    """Declared tier quality before an optional trusted-judge blend."""
    return result.ranking_score if result.ranking_score is not None else result.objective_score


def headline_quality(
    result: ModelResult, judge_trusted: bool, weight_judge: float = DEFAULT_WEIGHT_JUDGE
) -> float:
    """Blend tier base quality + judge when trusted; base quality alone otherwise."""
    base = base_quality(result)
    if judge_trusted and result.judge_score is not None:
        return (1.0 - weight_judge) * base + weight_judge * result.judge_score
    return base


def _vram_key(result: ModelResult) -> float:
    return result.peak_vram_mb if result.peak_vram_mb is not None else float("inf")


def rank_results(
    results: list[ModelResult],
    judge_trusted: bool = False,
    weight_judge: float = DEFAULT_WEIGHT_JUDGE,
) -> list[LeaderboardRow]:
    """Return ranked row dicts. Feasible models ranked by quality; infeasible appended."""
    feasible = [r for r in results if r.feasible]
    infeasible = [r for r in results if not r.feasible]
    ordered = sorted(
        feasible,
        key=lambda r: (
            -headline_quality(r, judge_trusted, weight_judge),
            -r.tokens_per_s,
            _vram_key(r),
        ),
    )
    rows: list[LeaderboardRow] = []
    for rank, r in enumerate(ordered, 1):
        rows.append(_row(r, rank, judge_trusted, weight_judge))
    for r in infeasible:
        rows.append(_row(r, None, judge_trusted, weight_judge))
    return rows


def _row(
    r: ModelResult, rank: int | None, judge_trusted: bool, weight_judge: float
) -> LeaderboardRow:
    row: LeaderboardRow = {
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
    if r.token_precision is not None:
        row["token_precision"] = round(r.token_precision, 4)
    if r.token_recall is not None:
        row["token_recall"] = round(r.token_recall, 4)
    if r.found_rate is not None:
        row["found_rate"] = round(r.found_rate, 4)
    if r.mean_completion_tokens is not None:
        row["mean_completion_tokens"] = round(r.mean_completion_tokens, 1)
    return row


def _table_cell(row: LeaderboardRow, key: str) -> str:
    mapping = {
        "rank": "-" if row["rank"] is None else str(row["rank"]),
        "model": row["model"],
        "backend": row["backend"],
        "quality": f"{row['quality']:.3f}",
        "objective": f"{row['objective']:.3f}",
        "precision": (
            "-" if row.get("token_precision") is None else f"{row['token_precision']:.3f}"
        ),
        "recall": "-" if row.get("token_recall") is None else f"{row['token_recall']:.3f}",
        "found": "-" if row.get("found_rate") is None else f"{row['found_rate']:.3f}",
        "len": (
            "-"
            if row.get("mean_completion_tokens") is None
            else f"{row['mean_completion_tokens']:.1f}"
        ),
        "judge": "-" if row.get("judge") is None else f"{row['judge']:.3f}",
        "reliab": f"{row['reliability']:.3f}",
        "tok/s": f"{row['tokens_per_s']:.1f}",
        "vram_mb": "-" if row["peak_vram_mb"] is None else f"{row['peak_vram_mb']:.0f}",
        "feasible": "yes" if row["feasible"] else "NO",
    }
    return mapping[key]


def format_table(rows: list[LeaderboardRow]) -> str:
    """Render ranked rows as an ASCII table (judge column omitted when always demoted)."""
    show_judge = any(row.get("judge") is not None for row in rows)
    headers = ["rank", "model", "backend", "quality", "objective"]
    if any(row.get("token_precision") is not None for row in rows):
        headers += ["precision", "recall", "found", "len"]
    if show_judge:
        headers.append("judge")
    headers += ["reliab", "tok/s", "vram_mb", "feasible"]

    table = [[_table_cell(row, h) for h in headers] for row in rows]

    widths = [
        max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    out = []
    if any(row.get("token_precision") is not None for row in rows):
        out.append(f"policy: quality = {policy_description()}; objective = token F1")
    out += [line, "  ".join("-" * widths[i] for i in range(len(headers)))]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)


def bootstrap_mean_ci(
    values: list[float], n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean of `values`. None for < 2 points."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_resamples))
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return (round(lo, 4), round(hi, 4))
