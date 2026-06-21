"""Rank model results into a leaderboard row (pure Python).

Ranking axis (design): GENERATION quality -- objective reference correctness, blended
with the gated judge ONLY when the judge is trusted, else objective alone. Fit/fail is a
hard filter (infeasible models are listed without a rank). Ties break by tokens/sec
(desc) then peak VRAM (asc), matching the Pareto tie-breaker order.

Milestone 1 produces a single row; the function is written for N so the multi-model
sweep in later milestones reuses it unchanged.
"""

import random
from collections import Counter
from dataclasses import dataclass, field

from llb.contracts import BoardRow, LeaderboardRow

DEFAULT_WEIGHT_JUDGE = 0.5
TIER_PRIVATE = "private"  # Tier-2 private gold-set metrics
TIER_SCREEN = "screen"  # Tier-1 public-screen metrics -- NEVER ranked against private ones


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
    feasible: bool = True
    tier: str = TIER_PRIVATE
    # Per-case score series -> bootstrap CIs (optional; aligned by case order).
    case_objectives: list[float] = field(default_factory=list)
    case_semantic: list[float] = field(default_factory=list)
    case_judge: list[float] = field(default_factory=list)


def per_case_quality(
    result: ModelResult, judge_trusted: bool, weight_judge: float = DEFAULT_WEIGHT_JUDGE
) -> list[float]:
    """The per-case HEADLINE quality series used for the rank-uncertainty CI: the trusted-judge
    blend per case when judge ratings are present, else the per-case objective scores."""
    if (
        judge_trusted
        and result.case_judge
        and len(result.case_judge) == len(result.case_objectives)
    ):
        return [
            (1.0 - weight_judge) * obj + weight_judge * jud
            for obj, jud in zip(result.case_objectives, result.case_judge)
        ]
    return list(result.case_objectives)


def headline_quality(
    result: ModelResult, judge_trusted: bool, weight_judge: float = DEFAULT_WEIGHT_JUDGE
) -> float:
    """Blend objective + judge when trusted; objective alone otherwise."""
    if judge_trusted and result.judge_score is not None:
        return (1.0 - weight_judge) * result.objective_score + weight_judge * result.judge_score
    return result.objective_score


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


def format_table(rows: list[LeaderboardRow]) -> str:
    """Render ranked rows as an ASCII table (judge column omitted when always demoted)."""
    show_judge = any(row.get("judge") is not None for row in rows)
    headers = ["rank", "model", "backend", "quality", "objective"]
    if show_judge:
        headers.append("judge")
    headers += ["reliab", "tok/s", "vram_mb", "feasible"]

    def cell(row: LeaderboardRow, key: str) -> str:
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
    widths = [
        max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    out = [line, "  ".join("-" * widths[i] for i in range(len(headers)))]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)


# --- Milestone 3.6: N-model rigor (average-rank, Pareto, confidence intervals) ------------


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


def _desc_ranks(scores: list[float]) -> list[float]:
    """1-based average ranks, higher score = better (rank 1); ties share the mean rank."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def quality_signals(results: list[ModelResult], judge_trusted: bool) -> list[str]:
    """Which quality axes every model shares: objective always; judge only when trusted +
    present for all; semantic only when present for all. The average-rank headline uses these,
    so it never rewards a model just for having an extra (missing-elsewhere) signal."""
    signals = ["objective_score"]
    if judge_trusted and all(r.judge_score is not None for r in results):
        signals.append("judge_score")
    if all(r.semantic_score is not None for r in results):
        signals.append("semantic_score")
    return signals


def average_ranks(results: list[ModelResult], judge_trusted: bool) -> dict[str, float]:
    """Mean of the per-quality-signal ranks for each model (lower is better)."""
    signals = quality_signals(results, judge_trusted)
    per_signal = {s: _desc_ranks([float(getattr(r, s) or 0.0) for r in results]) for s in signals}
    return {
        r.model: sum(per_signal[s][i] for s in signals) / len(signals)
        for i, r in enumerate(results)
    }


def pareto_front(results: list[ModelResult], judge_trusted: bool, weight_judge: float) -> set[str]:
    """Models not dominated on (quality up, tokens/sec up, peak VRAM down)."""

    def axes(r: ModelResult) -> tuple[float, float, float]:
        return (
            headline_quality(r, judge_trusted, weight_judge),
            r.tokens_per_s,
            -(_vram_key(r)),  # less VRAM is better -> negate so "higher is better" everywhere
        )

    front: set[str] = set()
    for r in results:
        ra = axes(r)
        dominated = any(
            other.model != r.model
            and all(o >= v for o, v in zip(axes(other), ra))
            and any(o > v for o, v in zip(axes(other), ra))
            for other in results
        )
        if not dominated:
            front.add(r.model)
    return front


def rank_board(
    results: list[ModelResult],
    *,
    judge_trusted: bool = False,
    weight_judge: float = DEFAULT_WEIGHT_JUDGE,
) -> list[BoardRow]:
    """N-model leaderboard: average-rank headline + weighted-blend + Pareto + CIs.

    Ordering is by average rank over the shared quality signals (robust to weight choice),
    tie-broken by the weighted-blend quality, then tok/s, then VRAM. Adjacent models whose
    objective CIs overlap are flagged `unresolved` (the flip is not statistically resolved).
    Refuses to mix Tier-1 screen and Tier-2 private metrics in one board.
    """
    model_counts = Counter(result.model for result in results)
    duplicates = sorted(model for model, count in model_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"board requires one selected config per model; duplicates: {duplicates}")
    tiers = {r.tier for r in results}
    if len(tiers) > 1:
        raise ValueError(f"cannot rank across tiers in one board: {sorted(tiers)}")

    feasible = [r for r in results if r.feasible]
    infeasible = [r for r in results if not r.feasible]
    # Reject an incompatible judge cohort: blending the judge for some models but not others
    # would compare a blended score against a bare objective. All-or-none when the judge is on.
    if judge_trusted:
        have_judge = [r.judge_score is not None for r in feasible]
        if any(have_judge) and not all(have_judge):
            raise ValueError(
                "incompatible judge cohort: judge trusted but some models lack a judge score"
            )
    avg = average_ranks(feasible, judge_trusted) if feasible else {}
    front = pareto_front(feasible, judge_trusted, weight_judge) if feasible else set()
    # The rank-uncertainty CI is over the per-case HEADLINE quality (blend when the judge is
    # trusted, else objective), so the overlap test compares what the ranking actually uses.
    cis = {
        r.model: bootstrap_mean_ci(per_case_quality(r, judge_trusted, weight_judge))
        for r in feasible
    }

    ordered = sorted(
        feasible,
        key=lambda r: (
            avg[r.model],
            -headline_quality(r, judge_trusted, weight_judge),
            -r.tokens_per_s,
            _vram_key(r),
        ),
    )

    rows: list[BoardRow] = []
    prev_ci: tuple[float, float] | None = None
    for rank, r in enumerate(ordered, 1):
        ci = cis.get(r.model)
        unresolved = bool(ci and prev_ci and ci[1] >= prev_ci[0])
        rows.append(
            _board_row(
                r, rank, avg[r.model], ci, r.model in front, unresolved, judge_trusted, weight_judge
            )
        )
        prev_ci = ci
    for r in infeasible:
        rows.append(
            _board_row(r, None, float("inf"), None, False, False, judge_trusted, weight_judge)
        )
    return rows


def _board_row(
    r: ModelResult,
    rank: int | None,
    avg_rank: float,
    ci: tuple[float, float] | None,
    pareto: bool,
    unresolved: bool,
    judge_trusted: bool,
    weight_judge: float,
) -> BoardRow:
    row: BoardRow = {
        "rank": rank,
        "model": r.model,
        "backend": r.backend,
        "tier": r.tier,
        "quality": round(headline_quality(r, judge_trusted, weight_judge), 4),
        "avg_rank": round(avg_rank, 3) if avg_rank != float("inf") else avg_rank,
        "objective": round(r.objective_score, 4),
        "judge": None if r.judge_score is None else round(r.judge_score, 4),
        "semantic": None if r.semantic_score is None else round(r.semantic_score, 4),
        "reliability": round(r.reliability, 4),
        "tokens_per_s": round(r.tokens_per_s, 2),
        "peak_vram_mb": r.peak_vram_mb,
        "pareto": pareto,
        "unresolved": unresolved,
        "feasible": r.feasible,
        "n_cases": r.n_cases,
    }
    if ci is not None:
        row["quality_ci"] = ci
    # Per-signal CIs are recorded only when that signal's per-case series is present, so the
    # board never invents a CI for a signal a model did not actually produce.
    objective_ci = bootstrap_mean_ci(r.case_objectives)
    if objective_ci is not None:
        row["objective_ci"] = objective_ci
    semantic_ci = bootstrap_mean_ci(r.case_semantic)
    if semantic_ci is not None:
        row["semantic_ci"] = semantic_ci
    if judge_trusted:
        judge_ci = bootstrap_mean_ci(r.case_judge)
        if judge_ci is not None:
            row["judge_ci"] = judge_ci
    return row


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


def format_board(rows: list[BoardRow], policy: str | None = None) -> str:
    """ASCII board: rank, avg-rank, quality (with CI + '~' for unresolved), Pareto star.
    Pass `policy` (see `ranking_policy_note`) to print the ranking method above the table."""
    headers = ["rank", "model", "backend", "avg_rank", "quality", "ci", "tok/s", "vram_mb", "P"]

    def cell(row: BoardRow, key: str) -> str:
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

    table = [[cell(row, h) for h in headers] for row in rows]
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
