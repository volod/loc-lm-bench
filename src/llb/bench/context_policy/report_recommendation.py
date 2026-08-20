"""Policy table and recommendation rendering for agent context comparisons."""

from llb.bench.context_policy.report import (
    BASELINE_POLICY,
    METRIC_COMPLETION,
    METRIC_PROMPT_TOKENS,
    PolicyReport,
    completion_reading,
)
from llb.prompts.registry import render_text
from llb.rag.fusion_evidence.evidence_gate import READING_SEPARATED, reading_label

RECOMMENDATION_TEMPLATE = "bench.agentic.context_recommendation"


def _delta_cell(report: PolicyReport, metric: str) -> str:
    comparison = report.paired.get(metric)
    if comparison is None:
        return "baseline"
    delta = comparison["delta"]
    return f"{delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]"


def format_policy_table(reports: list[PolicyReport]) -> str:
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
    if not reports:
        raise ValueError("cannot pick a context-policy winner from an empty report list")
    baseline = next((report for report in reports if report.policy == BASELINE_POLICY), None)
    if baseline is None:
        return min(
            reports,
            key=lambda report: (
                -report.result.objective_score,
                report.metric_mean(METRIC_PROMPT_TOKENS),
            ),
        )
    separated = [
        report
        for report in reports
        if report.paired
        and completion_reading(report) == READING_SEPARATED
        and report.paired[METRIC_COMPLETION]["delta"]["mean"] > 0.0
    ]
    if separated:
        return max(
            separated,
            key=lambda report: report.paired[METRIC_COMPLETION]["delta"]["mean"],
        )
    affordable = [
        report
        for report in reports
        if report.n_context_overflow <= baseline.n_context_overflow
        and report.metric_mean(METRIC_PROMPT_TOKENS) > 0
    ]
    return (
        min(affordable, key=lambda report: report.metric_mean(METRIC_PROMPT_TOKENS))
        if affordable
        else baseline
    )


def build_recommendation(model: str, reports: list[PolicyReport]) -> str:
    winner = _winner(reports)
    flat = not any(
        completion_reading(report) == READING_SEPARATED for report in reports if report.paired
    )
    ranking = ", ".join(
        f"{report.policy} {report.result.objective_score:.3f} "
        f"({report.metric_mean(METRIC_PROMPT_TOKENS):.0f} tok)"
        for report in reports
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
