"""Optuna trial construction and progressive pruning for multi-objective tuning."""

import statistics
from collections.abc import Callable, Sequence
from typing import Any

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.optimize.objectives import TrialMetrics, metrics_vector, normalize_outcome
from llb.optimize.tuner_runtime import TrialCallback
from llb.optimize.tuning_space import (
    estimate_prompt_tokens,
    fits_context,
    is_oom,
    suggest_overrides,
)

PRUNE_FRACTIONS = (0.25, 0.5)
PRUNE_WARMUP_TRIALS = 5
PRUNE_ATTR_PREFIX = "prune_quality_step_"


def make_multi_objective(
    base_config: RunConfig,
    evaluate: Callable[..., Any],
    objectives: Sequence[str],
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
    on_trial: TrialCallback | None = None,
    strategies: list[str] | None = None,
    reranker: str | None = None,
    embedders: Sequence[str] | None = None,
    tune_context_budget: bool = True,
    prune_case_count: int | None = None,
) -> Callable[[Any], tuple[float, ...]]:
    """Build an NSGA-II objective: sample, validate, prune, then evaluate."""
    import optuna

    def objective(trial: Any) -> tuple[float, ...]:
        overrides = suggest_overrides(
            trial,
            backend=base_config.backend,
            strategies=strategies,
            reranker=reranker,
            embedders=embedders,
            tune_context_budget=tune_context_budget,
        )
        try:
            config = base_config.with_overrides(**overrides)
        except ValueError as exc:
            raise optuna.TrialPruned(f"invalid config: {exc}") from None
        if not fits_context(config, model_spec, vram_mib, ram_mib):
            raise optuna.TrialPruned(
                f"retrieved context ~{estimate_prompt_tokens(config)} tok exceeds the budget/window"
            )
        trial.set_user_attr("overrides", overrides)
        try:
            metrics = _evaluate_with_pruning(
                trial, evaluate, config, prune_case_count=prune_case_count
            )
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            if is_oom(exc):
                raise optuna.TrialPruned(f"measured OOM: {exc}") from None
            raise
        _record_metrics(trial, metrics, overrides, on_trial)
        return metrics_vector(metrics, objectives)

    return objective


def _record_metrics(
    trial: Any,
    metrics: TrialMetrics,
    overrides: dict[str, Any],
    on_trial: TrialCallback | None,
) -> None:
    trial.set_user_attr("throughput", metrics.throughput)
    trial.set_user_attr("latency_s", metrics.latency_s)
    trial.set_user_attr("cost_usd", metrics.cost_usd)
    if on_trial is not None:
        on_trial(
            {
                "number": trial.number,
                "quality": metrics.quality,
                "latency_s": metrics.latency_s,
                "cost_usd": metrics.cost_usd,
                "throughput": metrics.throughput,
                **overrides,
            }
        )


def _evaluate_with_pruning(
    trial: Any,
    evaluate: Callable[..., Any],
    config: RunConfig,
    *,
    prune_case_count: int | None,
) -> TrialMetrics:
    """Median-style early stop on progressive subsets, then run the full evaluation."""
    import optuna

    if prune_case_count and prune_case_count > 1:
        for step, fraction in enumerate(PRUNE_FRACTIONS):
            limit = max(1, int(prune_case_count * fraction))
            partial = _call_evaluate(evaluate, config, limit=limit)
            trial.set_user_attr(f"{PRUNE_ATTR_PREFIX}{step}", partial.quality)
            if _below_step_median(trial, step, partial.quality):
                raise optuna.TrialPruned(
                    f"median-pruned at subset limit={limit} quality={partial.quality:.4f}"
                )
    return _call_evaluate(evaluate, config, limit=None)


def _below_step_median(trial: Any, step: int, quality: float) -> bool:
    """Return whether this subset quality trails enough prior trials' median."""
    key = f"{PRUNE_ATTR_PREFIX}{step}"
    prior = [
        float(other.user_attrs[key])
        for other in trial.study.trials
        if other.number != trial.number and key in other.user_attrs
    ]
    return len(prior) >= PRUNE_WARMUP_TRIALS and quality < statistics.median(prior)


def _call_evaluate(
    evaluate: Callable[..., Any], config: RunConfig, *, limit: int | None
) -> TrialMetrics:
    try:
        outcome = evaluate(config, limit=limit)
    except TypeError:
        outcome = evaluate(config)
    return normalize_outcome(outcome)
