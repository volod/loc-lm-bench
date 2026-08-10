"""Persistence projections for agent context-policy reports."""

from typing import Any

from llb.bench.agentic.context_policy import CONTEXT_POLICIES
from llb.bench.agentic_context_report import (
    METHOD,
    METRIC_PROMPT_TOKENS,
    METRIC_TOTAL_MODEL_INPUT_TOKENS,
    PolicyReport,
)
from llb.core.contracts.runs import RunMetrics
from llb.scoring.aggregate import TIER_AGENTIC


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
    return {
        "objective_score": round(report.result.objective_score, 6),
        "reliability": report.reliability,
        "tokens_per_s": report.result.tokens_per_s,
    }


def known_policies(policies: list[str]) -> list[str]:
    unknown = [policy for policy in policies if policy not in CONTEXT_POLICIES]
    if unknown:
        raise SystemExit(f"unknown context policies: {unknown}; choose from {CONTEXT_POLICIES}")
    return policies
