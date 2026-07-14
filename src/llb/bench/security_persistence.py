"""Focused security persistence implementation."""

from typing import Any
from llb.bench.common import (
    category_result,
)
from llb.core.contracts import (
    JudgeStatus,
    RunMetrics,
    RunPaths,
)
from llb.scoring import security
from llb.scoring.aggregate import TIER_SECURITY
from llb.scoring.leaderboard import ModelResult, bootstrap_mean_ci
from llb.bench.security_scoring import (
    _LOG,
    _RefusalJudgeRun,
    _ScoredSecurityCases,
    _SecurityJudgeConfig,
)


def _security_result(
    model: str, backend: str, scored: _ScoredSecurityCases, tokens_per_s: float
) -> ModelResult:
    return category_result(
        model=model,
        backend=backend,
        tier=TIER_SECURITY,
        case_objectives=scored.score.case_defended,  # per-attack-case defended -> defense-rate CI
        reliability=scored.reliability,
        tokens_per_s=tokens_per_s,
    )


def _security_metrics(result: ModelResult, reliability: float, tokens_per_s: float) -> RunMetrics:
    return {
        "objective_score": result.objective_score,  # defense rate (1 - ASR)
        "reliability": reliability,
        "tokens_per_s": tokens_per_s,
    }


def _score_config(
    model: str,
    backend: str,
    score: security.SecurityScore,
    asr_ci: tuple[float, float] | None,
    refusal_ci: tuple[float, float] | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "backend": backend,
        "tier": TIER_SECURITY,
        "category": "security",
        "n_cases": score.n_cases,
        "n_attacks": score.n_attacks,
        "asr": score.asr,
        "defense_rate": score.defense_rate,
        "refusal_appropriateness": score.refusal_appropriateness,
        "asr_by_family": score.asr_by_family,
        "defense_ci": list(asr_ci) if asr_ci else None,
        "refusal_appropriateness_ci": list(refusal_ci) if refusal_ci else None,
        "cross_language": _cross_language_config(score.cross_language),
        "bias_pairs": _bias_pair_config(score.bias_pairs),
    }


def _cross_language_config(
    xlang: security.CrossLanguageConsistency | None,
) -> dict[str, Any] | None:
    """Persist the cross-language-consistency block (None when the set has no matched groups)."""
    if xlang is None:
        return None
    ci = bootstrap_mean_ci(xlang.group_consistent)
    return {
        "n_groups": xlang.n_groups,
        "consistency": xlang.consistency,
        "refusal_rate_by_lang": xlang.refusal_rate_by_lang,
        "consistency_ci": list(ci) if ci else None,
    }


def _bias_pair_config(
    bias: security.BiasPairConsistency | None,
) -> dict[str, Any] | None:
    """Persist the matched-pair bias-consistency block (None when the set has no matched pairs)."""
    if bias is None:
        return None
    ci = bootstrap_mean_ci(bias.group_consistent)
    return {
        "n_pairs": bias.n_pairs,
        "consistency": bias.consistency,
        "refusal_rate_by_variant": bias.refusal_rate_by_variant,
        "consistency_ci": list(ci) if ci else None,
    }


def _judge_config(judge: _RefusalJudgeRun) -> dict[str, Any]:
    return {
        "judge_trusted": judge.outcome.trusted,
        "refusal_quality": judge.quality,  # gated diagnostic, NOT the headline
        "refusal_quality_ci": list(judge.quality_ci) if judge.quality_ci else None,
        "judge_diagnostics": judge.outcome.diagnostics,
    }


def _judge_status(config: _SecurityJudgeConfig, judge: _RefusalJudgeRun) -> JudgeStatus | None:
    if config.model is None:
        return None
    return {
        "calibration_rho": config.rho,
        "threshold": config.threshold,
        "trusted": judge.outcome.trusted,
        "model": config.model,
        "metrics": ["refusal_quality"],
        "diagnostics": judge.outcome.diagnostics,
    }


def _log_persisted_run(
    model: str,
    score: security.SecurityScore,
    judge: _RefusalJudgeRun,
    paths: RunPaths,
    tokens_per_s: float,
) -> None:
    xlang = (
        f"{score.cross_language.consistency:.3f} ({score.cross_language.n_groups}g)"
        if score.cross_language is not None
        else "n/a"
    )
    bias = (
        f"{score.bias_pairs.consistency:.3f} ({score.bias_pairs.n_pairs}p)"
        if score.bias_pairs is not None
        else "n/a"
    )
    _LOG.info(
        "[security] %s ASR=%.3f defense=%.3f refusal-appropriateness=%.3f "
        "xlang-consistency=%s bias-consistency=%s quality=%s tok/s=%.1f -> %s",
        model,
        score.asr,
        score.defense_rate,
        score.refusal_appropriateness,
        xlang,
        bias,
        f"{judge.quality:.3f}" if judge.quality is not None else "n/a",
        tokens_per_s,
        paths["manifest"],
    )
