"""Paired evidence and recommendation rendering for the agent-loop policy grid."""

from dataclasses import dataclass, field
from typing import cast

from llb.bench.agentic.model import AgenticRun, Episode
from llb.bench.agentic.loop_policy import LoopPolicy
from llb.bench.common import mean
from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.core.contracts.results import BoardRow
from llb.core.contracts.runs import RunPaths
from llb.rag.fusion_evidence.evidence_gate import reading_label
from llb.rag.fusion_evidence.paired import (
    PairedComparison,
    paired_comparison,
    reading_of,
)
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED
from llb.rag.fusion_evidence.stats import bootstrap_index_sets

METHOD = "agentic-loop-policy"

METRIC_COMPLETION = "completion"
METRIC_MALFORMED_RATE = "malformed_call_rate"
METRIC_STEPS = "n_steps"
METRIC_TOOL_CALLS = "n_tool_calls"
METRIC_PROMPT_TOKENS = "total_model_input_tokens"
METRIC_WALL_CLOCK = "elapsed_s"
METRICS = (
    METRIC_COMPLETION,
    METRIC_MALFORMED_RATE,
    METRIC_STEPS,
    METRIC_TOOL_CALLS,
    METRIC_PROMPT_TOKENS,
    METRIC_WALL_CLOCK,
)

BASELINE_MAX_STEPS = 6
BASELINE_POLICY = LoopPolicy()


@dataclass(frozen=True, slots=True)
class LoopPolicyCell:
    max_steps: int
    policy: LoopPolicy

    @property
    def cell_id(self) -> str:
        return (
            f"steps={self.max_steps},malformed={self.policy.malformed_call},"
            f"repeat={self.policy.repeated_call}"
        )

    @property
    def is_baseline(self) -> bool:
        return self.max_steps == BASELINE_MAX_STEPS and self.policy == BASELINE_POLICY


@dataclass(slots=True)
class LoopPolicyReport:
    cell: LoopPolicyCell
    run: AgenticRun
    paired: dict[str, PairedComparison] = field(default_factory=dict)
    paths: RunPaths | None = None

    @property
    def rows(self) -> list[AgenticCaseRow]:
        return self.run.rows

    @property
    def episodes(self) -> list[Episode]:
        return self.run.episodes

    def vector(self, metric: str) -> list[float]:
        if metric == METRIC_COMPLETION:
            return [float(row["success"]) for row in self.rows]
        return [float(cast(float, row.get(metric, 0.0))) for row in self.rows]

    def metric_mean(self, metric: str) -> float:
        return mean(self.vector(metric))

    @property
    def malformed_rate(self) -> float:
        attempts = sum(ep.n_steps + ep.n_repair_attempts for ep in self.episodes)
        return sum(ep.n_malformed_calls for ep in self.episodes) / max(attempts, 1)


@dataclass(slots=True)
class AgenticLoopPolicyRun:
    model: str
    backend: str
    reports: list[LoopPolicyReport]
    board: list[BoardRow]
    table: str
    recommendation: dict[str, object]
    task_set_digest: str


def pair_reports(
    reports: list[LoopPolicyReport],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> None:
    """Pair every cell, including the zero-delta baseline, against the baseline cell."""
    baseline = next((report for report in reports if report.cell.is_baseline), None)
    if baseline is None:
        raise ValueError("loop-policy grid has no steps=6,answer,allow baseline")
    indexes = bootstrap_index_sets(len(baseline.rows), resamples, seed)
    for report in reports:
        report.paired = {
            metric: paired_comparison(
                report.vector(metric),
                baseline.vector(metric),
                indexes,
                confidence,
            )
            for metric in METRICS
        }


def _delta(report: LoopPolicyReport, metric: str) -> str:
    interval = report.paired[metric]["delta"]
    return f"{interval['mean']:+.3f} [{interval['lo']:+.3f},{interval['hi']:+.3f}]"


def format_policy_table(reports: list[LoopPolicyReport]) -> str:
    """Completion, formatting failures, cost, and paired baseline deltas for every cell."""
    header = (
        f"{'max':>3} {'malformed':<11} {'repeat':<6} {'complete':>8} {'bad-rate':>8} "
        f"{'steps':>6} {'calls':>6} {'prompt-tok':>10} {'wall-s':>8} {'d(complete)':<23} "
        f"{'d(prompt)':<23} reading"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        completion_pair = report.paired[METRIC_COMPLETION]
        lines.append(
            f"{report.cell.max_steps:>3d} "
            f"{report.cell.policy.malformed_call:<11} "
            f"{report.cell.policy.repeated_call:<6} "
            f"{report.run.result.objective_score:>8.3f} "
            f"{report.malformed_rate:>8.3f} "
            f"{report.metric_mean(METRIC_STEPS):>6.2f} "
            f"{report.metric_mean(METRIC_TOOL_CALLS):>6.2f} "
            f"{report.metric_mean(METRIC_PROMPT_TOKENS):>10.1f} "
            f"{report.metric_mean(METRIC_WALL_CLOCK):>8.2f} "
            f"{_delta(report, METRIC_COMPLETION):<23} "
            f"{_delta(report, METRIC_PROMPT_TOKENS):<23} "
            f"{reading_label(reading_of(completion_pair))}"
        )
    return "\n".join(lines)


def build_recommendation(model: str, reports: list[LoopPolicyReport]) -> dict[str, object]:
    """Change the baseline only for a positive completion delta under the standard verdict."""
    baseline = next(report for report in reports if report.cell.is_baseline)
    separated = [
        report
        for report in reports
        if not report.cell.is_baseline
        and report.paired[METRIC_COMPLETION]["delta"]["mean"] > 0.0
        and reading_of(report.paired[METRIC_COMPLETION]) == "separated"
    ]
    winner = min(
        separated,
        key=lambda report: (
            -report.run.result.objective_score,
            report.metric_mean(METRIC_PROMPT_TOKENS),
            report.metric_mean(METRIC_WALL_CLOCK),
        ),
        default=baseline,
    )
    completion_pair = winner.paired[METRIC_COMPLETION]
    changed = not winner.cell.is_baseline
    return {
        "model": model,
        "max_steps": winner.cell.max_steps,
        "malformed_call_policy": winner.cell.policy.malformed_call,
        "repeated_call_policy": winner.cell.policy.repeated_call,
        "changes_shipped_defaults": changed,
        "verdict": reading_of(completion_pair),
        "reason": (
            "positive paired completion delta clears the standard verdict"
            if changed
            else "no candidate has a positive paired completion delta under the standard verdict"
        ),
        "completion_rate": round(winner.run.result.objective_score, 6),
        "malformed_call_rate": round(winner.malformed_rate, 6),
        "mean_steps": round(winner.metric_mean(METRIC_STEPS), 4),
        "mean_tool_calls": round(winner.metric_mean(METRIC_TOOL_CALLS), 4),
        "mean_model_calls": round(winner.metric_mean("n_model_calls"), 4),
        "mean_total_model_input_tokens": round(winner.metric_mean(METRIC_PROMPT_TOKENS), 4),
        "mean_wall_clock_s": round(winner.metric_mean(METRIC_WALL_CLOCK), 4),
        "paired_completion_vs_baseline": completion_pair,
    }
