"""Persistence for agent-loop policy cells and their shared comparison artifacts."""

import logging
from pathlib import Path

from llb.artifacts.runs.members import (
    RunMember,
    study_analysis,
    study_design,
    table_report,
)
from llb.backends.context_budget import ContextBudget
from llb.bench.loop_policy.report import METHOD, LoopPolicyReport
from llb.bench.common import Mirror, persist_category_run

_LOG = logging.getLogger(__name__)


def persist_reports(
    reports: list[LoopPolicyReport],
    *,
    data_dir: Path | str,
    run_name: str,
    model: str,
    backend: str,
    task_digest: str,
    table: str,
    recommendation: dict[str, object],
    max_steps: list[int],
    malformed_policies: list[str],
    repeated_call_policies: list[str],
    repeated_feedback_variants: list[str],
    budget: ContextBudget,
    verification_config: dict[str, object],
    model_family: str | None,
    run_seed: int | None,
    repeat_power_design: dict[str, object] | None,
    repeat_power_analysis: dict[str, object] | None,
    repeat_feedback_design: dict[str, object] | None,
    repeat_feedback_analysis: dict[str, object] | None,
    mirror: Mirror | None,
) -> None:
    """Write one atomic category bundle per cell, each carrying the shared comparison."""
    declared = (
        ("study-design.json", study_design, repeat_power_design),
        ("feedback-study-design.json", study_design, repeat_feedback_design),
        ("power-analysis.json", study_analysis, repeat_power_analysis),
        ("feedback-analysis.json", study_analysis, repeat_feedback_analysis),
    )
    artifacts: list[RunMember] = [
        table_report("comparison.md", "Agent loop policy comparison", table),
        study_analysis("recommendation.json", recommendation),
        *(build(name, payload) for name, build, payload in declared if payload is not None),
    ]
    for report in reports:
        config: dict[str, object] = {
            "model": model,
            "backend": backend,
            "tier": report.run.result.tier,
            "category": METHOD,
            "cell_id": report.cell.cell_id,
            "max_steps": report.cell.max_steps,
            "malformed_call_policy": report.cell.policy.malformed_call,
            "repeated_call_policy": report.cell.policy.repeated_call,
            "repeat_feedback_variant": report.cell.policy.repeat_feedback,
            "is_baseline": report.cell.is_baseline,
            "task_set_digest": task_digest,
            "model_family": model_family,
            "run_seed": run_seed,
            "n_tasks": len(report.rows),
            "grid": {
                "max_steps": max_steps,
                "malformed_call_policy": malformed_policies,
                "repeated_call_policy": repeated_call_policies,
                "repeat_feedback_variant": repeated_feedback_variants,
            },
            "completion_rate": round(report.run.result.objective_score, 6),
            "malformed_call_rate": round(report.malformed_rate, 6),
            "mean_steps": round(report.metric_mean("n_steps"), 4),
            "mean_tool_calls": round(report.metric_mean("n_tool_calls"), 4),
            "repeat_activation_rate": round(report.repeat_activation_rate, 6),
            "mean_repeated_calls": round(report.metric_mean("n_repeated_calls"), 4),
            "mean_model_calls": round(report.metric_mean("n_model_calls"), 4),
            "mean_total_model_input_tokens": round(
                report.metric_mean("total_model_input_tokens"), 4
            ),
            "mean_wall_clock_s": round(report.metric_mean("elapsed_s"), 4),
            "paired_vs_baseline": report.paired,
            "task_family_counts": (
                (repeat_feedback_analysis or repeat_power_analysis or {}).get(
                    "task_family_counts", {}
                )
            ),
            "repeat_power": repeat_power_analysis,
            "repeat_feedback": repeat_feedback_analysis,
            "recommendation": recommendation,
            **budget.provenance(),
            **verification_config,
        }
        report.paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=f"{run_name}-{report.cell.cell_id}",
            config=config,
            metrics={
                "objective_score": round(report.run.result.objective_score, 6),
                "reliability": report.run.result.reliability,
                "tokens_per_s": report.run.result.tokens_per_s,
            },
            case_rows=report.rows,
            mirror=mirror,
            artifacts=artifacts,
        )
        _LOG.info(
            "[agentic-loop-policy] %s completion=%.3f malformed=%.3f -> %s",
            report.cell.cell_id,
            report.run.result.objective_score,
            report.malformed_rate,
            report.paths["manifest"],
        )
