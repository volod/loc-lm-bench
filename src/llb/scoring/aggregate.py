"""Rank model results into a leaderboard row (pure Python).

Ranking axis (design): GENERATION quality -- objective reference correctness, blended
with the gated judge ONLY when the judge is trusted, else objective alone. Fit/fail is a
hard filter (infeasible models are listed without a rank). Ties break by tokens/sec
(desc) then peak VRAM (asc), matching the Pareto tie-breaker order.

RAG core produces a single row; the function is written for N so the multi-model
sweep in later current-state topics reuses it unchanged.
"""

from collections import Counter

from llb.core.contracts import BoardRow
from llb.scoring.leaderboard import (
    DEFAULT_WEIGHT_JUDGE,
    ModelResult,
    _vram_key,
    bootstrap_mean_ci,
    headline_quality,
    per_case_quality,
)

TIER_SCREEN = "screen"  # Tier-1 public-screen metrics -- NEVER ranked against private ones
# category tiers. Each category is its own Tier and is NEVER cross-ranked with
# the RAG board or with another category: the `_validate_board_cohort` guard already refuses a
# board whose `ModelResult`s carry more than one distinct `tier`, so these constants are the
# named identities the category runners stamp onto their results.
TIER_TEXT_ANALYSIS = (
    "text_analysis"  # text-analysis and category expansion text-analysis (planted-label recovery)
)
TIER_SECURITY = (
    "security"  # security benchmark security / robustness (ASR + refusal-appropriateness)
)
TIER_TOOLING = (
    "tooling"  # tooling benchmark tooling / MCP / function-calling (call-only correctness)
)
TIER_AGENTIC = "agentic"  # agentic workflows (completion-rate + efficiency)
TIER_SUMMARIZATION = (
    "summarization"  # category expansion summarization (reference coverage + faithfulness)
)
TIER_STRUCTURED = "structured"  # structured output (schema conformance + field accuracy)
TIER_CHAIN_CONTEXT = (
    "chain_context"  # context-policy comparison for ONE model (policy is the ranked row label)
)


# --- ranking rigor: N-model rigor (average-rank, Pareto, confidence intervals) ------------


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


def _validate_board_cohort(results: list[ModelResult], judge_trusted: bool) -> None:
    """Refuse mixed tiers, duplicate models, or a partial judge cohort."""
    model_counts = Counter(result.model for result in results)
    duplicates = sorted(model for model, count in model_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"board requires one selected config per model; duplicates: {duplicates}")
    tiers = {r.tier for r in results}
    if len(tiers) > 1:
        raise ValueError(f"cannot rank across tiers in one board: {sorted(tiers)}")
    if not judge_trusted:
        return
    feasible = [r for r in results if r.feasible]
    have_judge = [r.judge_score is not None for r in feasible]
    if any(have_judge) and not all(have_judge):
        raise ValueError(
            "incompatible judge cohort: judge trusted but some models lack a judge score"
        )


def _partition_feasible(results: list[ModelResult]) -> tuple[list[ModelResult], list[ModelResult]]:
    feasible = [r for r in results if r.feasible]
    infeasible = [r for r in results if not r.feasible]
    return feasible, infeasible


def _quality_cis(
    feasible: list[ModelResult], judge_trusted: bool, weight_judge: float
) -> dict[str, tuple[float, float] | None]:
    """Per-model bootstrap CIs over the headline quality series used for ranking."""
    return {
        r.model: bootstrap_mean_ci(per_case_quality(r, judge_trusted, weight_judge))
        for r in feasible
    }


def _rank_feasible_rows(
    feasible: list[ModelResult],
    judge_trusted: bool,
    weight_judge: float,
    avg: dict[str, float],
    front: set[str],
    cis: dict[str, tuple[float, float] | None],
) -> list[BoardRow]:
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
    return rows


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
    _validate_board_cohort(results, judge_trusted)
    feasible, infeasible = _partition_feasible(results)
    avg = average_ranks(feasible, judge_trusted) if feasible else {}
    front = pareto_front(feasible, judge_trusted, weight_judge) if feasible else set()
    cis = _quality_cis(feasible, judge_trusted, weight_judge)
    rows = _rank_feasible_rows(feasible, judge_trusted, weight_judge, avg, front, cis)
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
