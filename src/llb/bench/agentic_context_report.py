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
from typing import cast

from llb.bench.agentic.context_policy import POLICY_FULL
from llb.bench.agentic.model import Episode
from llb.bench.common import mean
from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.core.contracts.results import BoardRow
from llb.core.contracts.runs import RunPaths
from llb.rag.fusion_evidence.evidence_gate import (
    DEFAULT_CONFIDENCE,
    READING_FLAT,
    READING_SEPARATED,
    apply_evidence_gate,
)
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets
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
METRIC_TOTAL_MODEL_INPUT_TOKENS = "total_model_input_tokens"
# The summarizer half of the model-input cost: what the compact policy paid to fold, on its own.
METRIC_COMPACTION_PROMPT_TOKENS = "compaction_prompt_tokens"
# What the summarize call was offered, and how much of it its input cap elided head-and-tail. The
# elided span is transcript the running summary was never shown, so a completion reading beside it
# says whether the cap trimmed evidence the summary needed.
METRIC_SUMMARY_INPUT_CHARS = "summary_input_chars"
METRIC_SUMMARY_INPUT_ELIDED_CHARS = "summary_input_elided_chars"
METRICS = (METRIC_COMPLETION, METRIC_STEPS, METRIC_TOOL_CALLS, METRIC_PROMPT_TOKENS)

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
    kind_table: str = ""
    aggregate_safe_verdict: str = ""


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
