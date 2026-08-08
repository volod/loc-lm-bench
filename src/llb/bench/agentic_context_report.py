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
from llb.bench.agentic.context_aggregate import task_kind
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
METRIC_TOTAL_MODEL_INPUT_TOKENS = "total_model_input_tokens"
# The summarizer half of the model-input cost: what the compact policy paid to fold, on its own.
METRIC_COMPACTION_PROMPT_TOKENS = "compaction_prompt_tokens"
# What the summarize call was offered, and how much of it its input cap elided head-and-tail. The
# elided span is transcript the running summary was never shown, so a completion reading beside it
# says whether the cap trimmed evidence the summary needed.
METRIC_SUMMARY_INPUT_CHARS = "summary_input_chars"
METRIC_SUMMARY_INPUT_ELIDED_CHARS = "summary_input_elided_chars"
METRICS = (METRIC_COMPLETION, METRIC_STEPS, METRIC_TOOL_CALLS, METRIC_PROMPT_TOKENS)

# Task-kind split: the generator's search-count / search-locate ids (plus seed "other").
KIND_COUNT = "count"
KIND_LOCATE = "locate"
KIND_OTHER = "other"
KIND_ORDER = (KIND_COUNT, KIND_LOCATE, KIND_OTHER)

# Pre-header count-slice completion on the 24-task 2026-07-28 evidence set: every search-count
# failed under observation_cap/compact (ran all steps, counted wrong) and under full/keep_last_n
# (overflow at step 1). Used so the kind table can state whether aggregate-safe trimming recovered
# the slice without requiring a second "legacy" policy row.
PRE_HEADER_COUNT_COMPLETION: dict[str, float] = {
    POLICY_FULL: 0.0,
    "observation_cap": 0.0,
    "keep_last_n": 0.0,
    "compact": 0.0,
}

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


def kind_indices(report: PolicyReport, kind: str) -> list[int]:
    """Task indexes whose `item_id` maps to `kind` (`count` / `locate` / `other`)."""
    return [
        i for i, row in enumerate(report.rows) if task_kind(str(row.get("item_id", ""))) == kind
    ]


def kind_completion(report: PolicyReport, kind: str) -> float | None:
    """Mean completion on one task kind; None when the set has no tasks of that kind."""
    indexes = kind_indices(report, kind)
    if not indexes:
        return None
    return mean([report.case_success[i] for i in indexes])


def kind_overflow(report: PolicyReport, kind: str) -> int:
    """How many tasks of `kind` ended as `context_overflow`."""
    from llb.bench.agentic.model import STATUS_CONTEXT_OVERFLOW

    return sum(
        1
        for i in kind_indices(report, kind)
        if report.rows[i].get("status") == STATUS_CONTEXT_OVERFLOW
    )


def pair_kind_completion(
    reports: list[PolicyReport],
    kind: str,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, PairedComparison]:
    """Paired completion deltas against `full` restricted to one task kind."""
    baseline = next((r for r in reports if r.policy == BASELINE_POLICY), None)
    if baseline is None:
        return {}
    indexes = kind_indices(baseline, kind)
    if len(indexes) < 2:
        return {}
    base_vec = [baseline.case_success[i] for i in indexes]
    index_sets = bootstrap_index_sets(len(indexes), resamples, seed)
    out: dict[str, PairedComparison] = {}
    for report in reports:
        if report.policy == BASELINE_POLICY:
            continue
        if len(report.case_success) != len(baseline.case_success):
            continue
        cand = [report.case_success[i] for i in indexes]
        out[report.policy] = paired_comparison(cand, base_vec, index_sets, confidence)
    return out


def _reference_report(reports: list[PolicyReport]) -> PolicyReport | None:
    """The report the kind breakdown is read against: the baseline policy, else the first run."""
    return next((row for row in reports if row.policy == BASELINE_POLICY), None) or (
        reports[0] if reports else None
    )


def _kind_header(present: list[str]) -> str:
    """The column head for the kinds this run actually has rows for."""
    header = f"{'policy':<16}" + "".join(f" {k:>10}" for k in present) + f" {'overflow-count':>14}"
    return header + f" {'vs-pre-header':>14}" if KIND_COUNT in present else header


def _kind_cells(report: PolicyReport, present: list[str]) -> str:
    """One policy's completion per kind, dashed where the kind produced no row."""
    cells = []
    for kind in present:
        value = kind_completion(report, kind)
        cells.append(f"{value:>10.3f}" if value is not None else f"{'-':>10}")
    return "".join(f" {cell}" for cell in cells)


def _pre_header_cell(report: PolicyReport, pre: dict[str, float]) -> str:
    """The count slice against the pre-header evidence: the recovery this table exists to show."""
    prior = pre.get(report.policy)
    now = kind_completion(report, KIND_COUNT)
    if prior is None or now is None:
        return f" {'-':>14}"
    return f" {now - prior:>+14.3f}"


def _kind_row(report: PolicyReport, present: list[str], pre: dict[str, float]) -> str:
    """One table row: the policy, its per-kind completion, its overflows, its recovery."""
    row = f"{report.policy:<16}" + _kind_cells(report, present)
    if KIND_COUNT not in present:
        return row
    return row + f" {kind_overflow(report, KIND_COUNT):>14d}" + _pre_header_cell(report, pre)


def _count_pair_lines(count_pairs: dict[str, PairedComparison]) -> list[str]:
    """The paired count-slice reading under the table, one line per policy."""
    lines = ["count-slice paired vs full:"]
    for policy, comparison in count_pairs.items():
        delta = comparison["delta"]
        reading = READING_SEPARATED if delta["lo"] > 0.0 or delta["hi"] < 0.0 else READING_FLAT
        lines.append(
            f"  {policy:<14} d(completion)={delta['mean']:+.3f} "
            f"[{delta['lo']:+.3f}, {delta['hi']:+.3f}] "
            f"w/l/t={comparison['wins']}/{comparison['losses']}/{comparison['ties']} "
            f"{reading_label(reading)}"
        )
    return lines


def format_kind_table(
    reports: list[PolicyReport],
    *,
    pre_header_count: dict[str, float] | None = None,
) -> str:
    """Per-policy completion broken out by task kind, plus count-slice vs pre-header."""
    reference = _reference_report(reports)
    if reference is None or not reference.rows:
        return ""
    present = [kind for kind in KIND_ORDER if kind_indices(reference, kind)]
    if not present:
        return ""
    pre = pre_header_count if pre_header_count is not None else PRE_HEADER_COUNT_COMPLETION
    header = _kind_header(present)
    lines = ["by task kind:", header, "-" * len(header)]
    lines.extend(_kind_row(report, present, pre) for report in reports)
    count_pairs = pair_kind_completion(reports, KIND_COUNT) if KIND_COUNT in present else {}
    if count_pairs:
        lines.extend(_count_pair_lines(count_pairs))
    return "\n".join(lines)


def _recovery_bit(name: str, prior: float, now: float) -> str:
    """One policy's count-slice movement against the pre-header evidence, read as prose."""
    delta = now - prior
    if delta > 0:
        return f"`{name}` count {prior:.3f}->{now:.3f} (recovered {delta:+.3f})"
    return f"`{name}` count {prior:.3f}->{now:.3f} (no recovery; loss is elsewhere or still flat)"


def _count_recoveries(
    reports: list[PolicyReport], pre: dict[str, float]
) -> list[tuple[str, float, float]]:
    """The trimming policies that produced a count-slice number this run, with what to compare."""
    measured: list[tuple[str, float, float]] = []
    for name in ("observation_cap", "compact"):
        report = next((row for row in reports if row.policy == name), None)
        now = kind_completion(report, KIND_COUNT) if report is not None else None
        if now is not None:
            measured.append((name, pre.get(name, 0.0), now))
    return measured


def aggregate_safe_verdict(
    reports: list[PolicyReport],
    *,
    pre_header_count: dict[str, float] | None = None,
) -> str:
    """Whether aggregate-safe trimming recovered the count slice vs the pre-header evidence."""
    pre = pre_header_count if pre_header_count is not None else PRE_HEADER_COUNT_COMPLETION
    reference = _reference_report(reports)
    if reference is None or not kind_indices(reference, KIND_COUNT):
        return "no count-slice tasks in this set; aggregate-safe trimming not scored"
    measured = _count_recoveries(reports, pre)
    if not measured:
        return "count slice present but observation_cap/compact were not in this run"
    bits = "; ".join(_recovery_bit(name, prior, now) for name, prior, now in measured)
    if any(now > prior for _name, prior, now in measured):
        return "aggregate-safe trimming recovered count-slice completion: " + bits
    return (
        "aggregate-safe trimming did NOT move the count slice vs the pre-header evidence: " + bits
    )


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
    the question the lane asks. When `full` is absent (a one-policy re-run), pick the best
    completion then the cheapest prompt among the reported rows.
    """
    if not reports:
        raise ValueError("cannot pick a context-policy winner from an empty report list")
    baseline = next((r for r in reports if r.policy == BASELINE_POLICY), None)
    if baseline is None:
        return min(
            reports,
            key=lambda r: (-r.result.objective_score, r.metric_mean(METRIC_PROMPT_TOKENS)),
        )
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
        "mean_total_model_input_tokens": round(
            report.metric_mean(METRIC_TOTAL_MODEL_INPUT_TOKENS), 4
        ),
        "mean_compaction_prompt_tokens": round(report.metric_mean("compaction_prompt_tokens"), 4),
        "mean_model_calls": round(report.metric_mean("n_model_calls"), 4),
        "mean_observation_bytes": round(report.metric_mean("observation_bytes"), 4),
        "n_compactions": int(sum(report.vector("n_compactions"))),
        "n_trimmed_observations": int(sum(report.vector("n_trimmed_observations"))),
        "n_context_overflow": report.n_context_overflow,
        "paired_vs_full": report.paired or None,
        "aggregate_safe": True,
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
