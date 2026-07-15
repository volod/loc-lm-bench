"""Persist one agentic run: assemble the manifest config + metrics + case rows + judge status and
write them through the shared `persist_category_run`.

Completion-rate is the headline metric; trajectory-quality (when trusted) rides along as a gated
diagnostic in the config payload, never in the headline.
"""

import logging

from llb.bench.agentic.model import METHOD, _AgenticPersistInput, _JudgeConfig
from llb.bench.common import persist_category_run
from llb.core.contracts.judging import JudgeStatus
from llb.core.contracts.runs import RunMetrics, RunPaths
from llb.scoring.aggregate import TIER_AGENTIC
from llb.scoring.leaderboard import ModelResult
from llb.scoring.judge.model import JudgeOutcome

_LOG = logging.getLogger(__name__)


def _agentic_metrics(result: ModelResult, reliability: float, tokens_per_s: float) -> RunMetrics:
    return {
        "objective_score": result.objective_score,  # completion rate
        "reliability": reliability,
        "tokens_per_s": tokens_per_s,
    }


def _agentic_config(request: _AgenticPersistInput) -> dict[str, object]:
    return {
        "model": request.model,
        "backend": request.backend,
        "tier": TIER_AGENTIC,
        "category": "agentic",
        "harness": request.harness_name,
        "prompt_system": request.prompt_system,
        "n_tasks": request.n_tasks,
        "max_steps": request.max_steps,
        "completion_rate": request.result.objective_score,
        "mean_trajectory_steps": round(request.scored.mean_steps, 4),
        "mean_tool_calls": round(request.scored.mean_tool_calls, 4),
        "completion_rate_ci": list(request.scored.completion_ci)
        if request.scored.completion_ci
        else None,
        "judge_trusted": request.quality.outcome.trusted,
        "trajectory_quality": request.quality.value,  # gated diagnostic, NOT the headline
        "trajectory_quality_ci": list(request.quality.ci) if request.quality.ci else None,
        "judge_diagnostics": request.quality.outcome.diagnostics,
        **request.verification_cfg,
    }


def _agentic_judge_status(
    config: _JudgeConfig,
    outcome: JudgeOutcome,
) -> JudgeStatus | None:
    if config.model is None:
        return None
    return {
        "calibration_rho": config.rho,
        "threshold": config.threshold,
        "trusted": outcome.trusted,
        "model": config.model,
        "metrics": ["trajectory_quality"],
        "diagnostics": outcome.diagnostics,
    }


def _persist_agentic_run(request: _AgenticPersistInput) -> RunPaths | None:
    if request.data_dir is None:
        return None
    paths = persist_category_run(
        method=METHOD,
        data_dir=request.data_dir,
        run_name=request.run_name,
        config=_agentic_config(request),
        metrics=_agentic_metrics(request.result, request.scored.reliability, request.tokens_per_s),
        case_rows=request.scored.rows,
        judge=_agentic_judge_status(request.judge_config, request.quality.outcome),
        mirror=request.mirror,
    )
    _LOG.info(
        "[agentic] %s completion=%.3f mean-steps=%.2f mean-tool-calls=%.2f quality=%s -> %s",
        request.model,
        request.result.objective_score,
        request.scored.mean_steps,
        request.scored.mean_tool_calls,
        f"{request.quality.value:.3f}" if request.quality.value is not None else "n/a",
        paths["manifest"],
    )
    return paths
