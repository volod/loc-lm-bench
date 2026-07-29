"""Result contract, paired reading, and rendering for the agent context-policy comparison.

The comparison holds the model, the task set, the tool world, and the success checks FIXED and
varies only the context-management policy, so every difference it reports is attributable to how
the loop spent its window. Four metrics travel per policy -- completion, steps, tool calls, and
prompt tokens -- each as a per-item vector paired against the `full` baseline over SHARED bootstrap
index sets, so an interval is about the DIFFERENCE and not about two lanes' separate sampling
noise. The statistics are reused wholesale from `llb.rag.fusion_evidence`; nothing here re-derives
them.
"""

from dataclasses import dataclass, field
from typing import Any, cast

from llb.bench.agentic.context import CONTEXT_POLICIES, POLICY_FULL
from llb.bench.agentic.model import Episode
from llb.bench.common import mean
from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.core.contracts.results import BoardRow
from llb.core.contracts.runs import RunMetrics, RunPaths
from llb.prompts.registry import render_text
from llb.rag.fusion_evidence.evidence_gate import (
    DEFAULT_CONFIDENCE,
    READING_FLAT,
    READING_SEPARATED,
    apply_evidence_gate,
    reading_label,
)
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets
from llb.scoring.aggregate import TIER_AGENTIC
from llb.scoring.leaderboard import ModelResult

METHOD = "agentic-context"

# The paired metrics. Completion is the headline and the only one a verdict is cut on; the other
# three are the COST of buying it, so a policy that completes as often on a smaller prompt is a
# better answer to "how should my agent spend its context window". Every metric name except
# `completion` is an `AgenticCaseRow` column, which is how `PolicyReport.vector` reads them.
METRIC_COMPLETION = "completion"
METRIC_STEPS = "n_steps"
METRIC_TOOL_CALLS = "n_tool_calls"
METRIC_PROMPT_TOKENS = "max_prompt_tokens"
METRICS = (METRIC_COMPLETION, METRIC_STEPS, METRIC_TOOL_CALLS, METRIC_PROMPT_TOKENS)

RECOMMENDATION_TEMPLATE = "bench.agentic.context_recommendation"

BASELINE_POLICY = POLICY_FULL


@dataclass(slots=True)
class PolicyReport:
    """One context policy's scored outcome over the whole task set."""

    policy: str
    result: ModelResult
    rows: list[AgenticCaseRow]
    episodes: list[Episode]
    case_success: list[float]
    reliability: float
    completion_ci: tuple[float, float] | None
    mean_steps: float
    mean_tool_calls: float
    n_context_overflow: int
    paired: dict[str, PairedComparison] = field(default_factory=dict)
    paths: RunPaths | None = None

    def vector(self, metric: str) -> list[float]:
        """The per-item metric vector this policy is paired on, in task order.

        A row missing an optional telemetry column reads 0.0 rather than dropping the item: the
        paired lane needs one value per task on BOTH sides or the delta is not a paired delta.
        """
        if metric == METRIC_COMPLETION:
            return list(self.case_success)
        return [float(cast(float, row.get(metric, 0))) for row in self.rows]

    def metric_mean(self, metric: str) -> float:
        return mean(self.vector(metric))


@dataclass(slots=True)
class AgenticContextRun:
    """Outcome of one context-policy comparison for a fixed model."""

    model: str
    backend: str
    reports: list[PolicyReport]
    board: list[BoardRow]
    table: str
    recommendation: str
    task_set_digest: str
    max_prompt_chars: int


def pair_against_baseline(
    reports: list[PolicyReport],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> None:
    """Attach every non-baseline policy's paired deltas against `full`, in place.

    One set of resample indices is drawn per comparison and shared across the four metrics, so a
    policy's completion delta and its prompt-token delta are read on the SAME resampled task sets.
    """
    baseline = next((r for r in reports if r.policy == BASELINE_POLICY), None)
    if baseline is None:
        return
    n_items = len(baseline.case_success)
    index_sets = bootstrap_index_sets(n_items, resamples, seed)
    for report in reports:
        if report.policy == BASELINE_POLICY or len(report.case_success) != n_items:
            continue
        report.paired = {
            metric: paired_comparison(
                report.vector(metric), baseline.vector(metric), index_sets, confidence
            )
            for metric in METRICS
        }


def completion_reading(report: PolicyReport, *, confidence: float = DEFAULT_CONFIDENCE) -> str:
    """`separated` / `flat` / `insufficient_evidence` for one policy's completion delta."""
    comparison = report.paired.get(METRIC_COMPLETION)
    if comparison is None:
        return READING_FLAT
    delta = comparison["delta"]
    reading = READING_SEPARATED if delta["lo"] > 0.0 or delta["hi"] < 0.0 else READING_FLAT
    return apply_evidence_gate(
        reading,
        discordant=comparison["wins"] + comparison["losses"],
        confidence=confidence,
    )


def _delta_cell(report: PolicyReport, metric: str) -> str:
    comparison = report.paired.get(metric)
    if comparison is None:
        return "baseline"
    delta = comparison["delta"]
    return f"{delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]"


def format_policy_table(reports: list[PolicyReport]) -> str:
    """The per-policy metric table: the four means, then the four paired deltas against `full`."""
    header = (
        f"{'policy':<16} {'completion':>10} {'steps':>7} {'calls':>7} {'prompt-tok':>11} "
        f"{'overflow':>9}  {'d(completion)':<24} {'d(prompt-tok)':<32} reading"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        lines.append(
            f"{report.policy:<16} "
            f"{report.result.objective_score:>10.3f} "
            f"{report.mean_steps:>7.2f} "
            f"{report.mean_tool_calls:>7.2f} "
            f"{report.metric_mean(METRIC_PROMPT_TOKENS):>11.1f} "
            f"{report.n_context_overflow:>9d}  "
            f"{_delta_cell(report, METRIC_COMPLETION):<24} "
            f"{_delta_cell(report, METRIC_PROMPT_TOKENS):<32} "
            f"{reading_label(completion_reading(report)) if report.paired else '-'}"
        )
    return "\n".join(lines)


def _winner(reports: list[PolicyReport]) -> PolicyReport:
    """The policy a reading would name: the separated completion gain, else the cheapest prompt.

    With no separated completion delta the policies are FLAT on the headline, and the honest pick
    is the one that reaches that same completion on the smallest prompt -- never a rank read off
    third-decimal noise. A policy that OVERFLOWS more often than the baseline is never named on the
    cost tie-break: it bought its cheap prompt by ending episodes early, which is the opposite of
    the question the lane asks.
    """
    baseline = next(r for r in reports if r.policy == BASELINE_POLICY)
    separated = [
        r
        for r in reports
        if r.paired
        and completion_reading(r) == READING_SEPARATED
        and r.paired[METRIC_COMPLETION]["delta"]["mean"] > 0.0
    ]
    if separated:
        return max(separated, key=lambda r: r.paired[METRIC_COMPLETION]["delta"]["mean"])
    affordable = [
        r
        for r in reports
        if r.n_context_overflow <= baseline.n_context_overflow
        and r.metric_mean(METRIC_PROMPT_TOKENS) > 0
    ]
    return (
        min(affordable, key=lambda r: r.metric_mean(METRIC_PROMPT_TOKENS))
        if affordable
        else baseline
    )


def build_recommendation(model: str, reports: list[PolicyReport]) -> str:
    """Name the best policy for this model, or state that the policies are flat at this task count."""
    winner = _winner(reports)
    flat = not any(completion_reading(r) == READING_SEPARATED for r in reports if r.paired)
    ranking = ", ".join(
        f"{r.policy} {r.result.objective_score:.3f} ({r.metric_mean(METRIC_PROMPT_TOKENS):.0f} tok)"
        for r in reports
    )
    return render_text(
        RECOMMENDATION_TEMPLATE,
        {
            "model": model,
            "winner": winner.policy,
            "winner_completion": f"{winner.result.objective_score:.3f}",
            "winner_prompt_tokens": f"{winner.metric_mean(METRIC_PROMPT_TOKENS):.0f}",
            "n_tasks": len(winner.case_success),
            "ranking": ranking,
            "verdict": _verdict(flat, winner),
        },
    )


def _verdict(flat: bool, winner: PolicyReport) -> str:
    if flat:
        return (
            "жодна політика не відокремилася від `full` за часткою завершення на цьому наборі "
            f"завдань — обирай `{winner.policy}` за вартістю контексту, а не за якістю"
        )
    return (
        f"`{winner.policy}` відокремилася від `full` за часткою завершення "
        f"({_delta_cell(winner, METRIC_COMPLETION)})"
    )


def policy_config(
    report: PolicyReport,
    *,
    model: str,
    backend: str,
    task_digest: str,
    policies: list[str],
    max_steps: int,
    max_prompt_chars: int,
    policy_settings: dict[str, Any],
    budget_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One persisted policy bundle's provenance configuration."""
    config: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "tier": TIER_AGENTIC,
        "category": METHOD,
        "policy": report.policy,
        "policies": policies,
        "policy_settings": policy_settings,
        "task_set_digest": task_digest,
        "n_tasks": len(report.case_success),
        "max_steps": max_steps,
        "max_prompt_chars": max_prompt_chars,
        "completion_rate": round(report.result.objective_score, 6),
        "completion_rate_ci": list(report.completion_ci) if report.completion_ci else None,
        "mean_trajectory_steps": round(report.mean_steps, 4),
        "mean_tool_calls": round(report.mean_tool_calls, 4),
        "mean_max_prompt_tokens": round(report.metric_mean(METRIC_PROMPT_TOKENS), 4),
        "mean_observation_bytes": round(report.metric_mean("observation_bytes"), 4),
        "n_compactions": int(sum(report.vector("n_compactions"))),
        "n_trimmed_observations": int(sum(report.vector("n_trimmed_observations"))),
        "n_context_overflow": report.n_context_overflow,
        "paired_vs_full": report.paired or None,
    }
    if budget_provenance:
        config.update(budget_provenance)
    return config


def policy_metrics(report: PolicyReport) -> RunMetrics:
    """Project a policy report into the common persisted metric contract."""
    return {
        "objective_score": round(report.result.objective_score, 6),
        "reliability": report.reliability,
        "tokens_per_s": report.result.tokens_per_s,
    }


def known_policies(policies: list[str]) -> list[str]:
    """Validate a requested policy list against the lane's vocabulary."""
    unknown = [p for p in policies if p not in CONTEXT_POLICIES]
    if unknown:
        raise SystemExit(f"unknown context policies: {unknown}; choose from {CONTEXT_POLICIES}")
    return policies
