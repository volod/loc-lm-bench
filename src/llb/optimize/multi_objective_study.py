"""Multi-objective Optuna tuning: NSGA-II + MedianPruner + Pareto report."""

from pathlib import Path
from typing import Any, Callable, Sequence

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.core.contracts.runs import EvalResult
from llb.optimize.objectives import OBJECTIVE_COST, parse_objectives
from llb.optimize.multi_objective_trial import make_multi_objective
from llb.optimize.multi_objective_runtime import (
    prepare_evaluate,
    prepare_store_registry,
    run_study,
    storage_url,
    study_outcome,
    write_study_report,
)
from llb.optimize.tuner_models import MultiObjectiveResult, MultiTwoStageResult
from llb.optimize.store_registry import StoreRegistry
from llb.optimize.tuner_runtime import TrialCallback, _LOG, _run_eval_final
from llb.optimize.tuning_space import FINAL_SPLIT


def tune_multi(
    base_config: RunConfig,
    *,
    n_trials: int,
    study_name: str,
    objectives: str | Sequence[str],
    evaluate: Callable[..., Any] | None = None,
    storage: str | None = None,
    seed: int = 13,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
    on_trial: TrialCallback | None = None,
    isolate: bool = False,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    gpu_sampler: Callable[[], list[Any]] | None = None,
    strategies: list[str] | None = None,
    reranker: str | None = None,
    embedders: Sequence[str] | None = None,
    tune_context_budget: bool = True,
    prune_case_count: int | None = None,
    accuracy_floor: float | None = None,
    report_dir: Any | None = None,
    write_report: bool = True,
    stores: StoreRegistry | None = None,
    prewarm_stores: bool = True,
) -> MultiObjectiveResult:
    """Run NSGA-II and emit Pareto picks, optionally using prewarmed embedder stores."""
    goals = parse_objectives(objectives)
    if OBJECTIVE_COST in goals and base_config.scorer_policy != "frontier":
        _LOG.warning(
            "[tune] cost objective requested but scorer_policy=%s; cost stays 0 unless the "
            "evaluate hook supplies it",
            base_config.scorer_policy,
        )
    stores = prepare_store_registry(
        base_config,
        study_name=study_name,
        embedders=embedders,
        evaluate=evaluate,
        stores=stores,
        prewarm=prewarm_stores,
    )
    evaluate_hook = prepare_evaluate(
        evaluate,
        stores,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        gpu_sampler=gpu_sampler,
    )
    storage = storage_url(base_config, study_name, storage)
    objective = make_multi_objective(
        base_config,
        evaluate_hook,
        goals,
        model_spec=model_spec,
        vram_mib=vram_mib,
        ram_mib=ram_mib,
        on_trial=on_trial,
        strategies=strategies,
        reranker=reranker,
        embedders=embedders,
        tune_context_budget=tune_context_budget,
        prune_case_count=prune_case_count,
    )
    study = run_study(
        study_name=study_name,
        storage=storage,
        goals=goals,
        seed=seed,
        objective=objective,
        n_trials=n_trials,
    )
    complete, pruned, front, picks = study_outcome(
        study,
        goals,
        study_name=study_name,
        accuracy_floor=accuracy_floor,
    )
    out_dir, paths = write_study_report(
        base_config,
        study_name=study_name,
        goals=goals,
        front=front,
        picks=picks,
        n_trials=len(study.trials),
        n_complete=len(complete),
        n_pruned=len(pruned),
        report_dir=report_dir,
        enabled=write_report,
    )
    _LOG.info(
        "[tune] %s multi-obj front=%d picks=%s over %d trials (%d pruned)",
        study_name,
        len(front),
        [p.goal for p in picks],
        len(study.trials),
        len(pruned),
    )
    return MultiObjectiveResult(
        study_name=study_name,
        storage=storage,
        objectives=goals,
        n_trials=len(study.trials),
        n_complete=len(complete),
        n_pruned=len(pruned),
        front=front,
        picks=picks,
        report_dir=out_dir,
        report_paths=paths,
        store_builds=list(stores.builds),
    )


def two_stage_multi(
    base_config: RunConfig,
    *,
    n_trials: int,
    study_name: str,
    objectives: str | Sequence[str],
    evaluate: Callable[..., Any] | None = None,
    final_runner: Callable[[RunConfig], EvalResult] | None = None,
    storage: str | None = None,
    seed: int = 13,
    model_spec: ModelSpec | None = None,
    vram_mib: int = 0,
    ram_mib: int = 0,
    on_trial: TrialCallback | None = None,
    isolate: bool = False,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    gpu_sampler: Callable[[], list[Any]] | None = None,
    strategies: list[str] | None = None,
    reranker: str | None = None,
    embedders: Sequence[str] | None = None,
    tune_context_budget: bool = True,
    prune_case_count: int | None = None,
    accuracy_floor: float | None = None,
    report_dir: Path | None = None,
    write_report: bool = True,
    stores: StoreRegistry | None = None,
    prewarm_stores: bool = True,
) -> MultiTwoStageResult:
    """Stage 1 multi-obj tune; stage 2 scores each named pick on the final split."""
    result = tune_multi(
        base_config,
        n_trials=n_trials,
        study_name=study_name,
        objectives=objectives,
        evaluate=evaluate,
        storage=storage,
        seed=seed,
        model_spec=model_spec,
        vram_mib=vram_mib,
        ram_mib=ram_mib,
        on_trial=on_trial,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        gpu_sampler=gpu_sampler,
        strategies=strategies,
        reranker=reranker,
        embedders=embedders,
        tune_context_budget=tune_context_budget,
        prune_case_count=prune_case_count,
        accuracy_floor=accuracy_floor,
        report_dir=report_dir,
        write_report=write_report,
        stores=stores,
        prewarm_stores=prewarm_stores,
    )
    runner = final_runner or _run_eval_final
    finals: dict[str, EvalResult] = {}
    for pick in result.picks:
        cfg = result.config_for(base_config, pick.goal)
        _LOG.info(
            "[tune] %s stage-2 scoring pick=%s on the '%s' split",
            study_name,
            pick.goal,
            FINAL_SPLIT,
        )
        finals[pick.goal] = runner(cfg)
    return MultiTwoStageResult(tune=result, finals=finals)
