"""Study setup, execution, and result projection for multi-objective tuning."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from llb.core.config import RunConfig
from llb.optimize.objectives import (
    OBJECTIVE_COST,
    extract_pareto_front,
    nondominated_trials,
    select_goal_picks,
    study_directions,
)
from llb.optimize.pareto_report import tune_run_dir, write_pareto_report
from llb.optimize.store_registry import StoreRegistry, study_stores_dir
from llb.optimize.tuner_runtime import _run_eval_metrics
from llb.optimize.tuning_space import with_isolation

OPTUNA_METHOD = "optuna"


def prepare_store_registry(
    base_config: RunConfig,
    *,
    study_name: str,
    embedders: Sequence[str] | None,
    evaluate: Callable[..., Any] | None,
    stores: StoreRegistry | None,
    prewarm: bool,
) -> StoreRegistry:
    """Build or configure the registry and prewarm it only when evaluation consumes it."""
    shared_registry = stores is not None
    registry = stores or StoreRegistry(
        cache_dir=_store_cache_dir(base_config, study_name), embedders=embedders
    )
    if embedders and registry.embedders is None:
        registry.embedders = list(embedders)
    if prewarm and embedders and (evaluate is None or shared_registry):
        registry.prewarm(base_config, embedders)
    return registry


def _store_cache_dir(base_config: RunConfig, study_name: str) -> Path | None:
    if not study_name:
        return None
    cache_dir = study_stores_dir(base_config.data_dir, study_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def prepare_evaluate(
    evaluate: Callable[..., Any] | None,
    stores: StoreRegistry,
    *,
    isolate: bool,
    vram_reader: Callable[[], int] | None,
    pid_usage_reader: Callable[[], dict[int, int]] | None,
    gpu_sampler: Callable[[], list[Any]] | None,
) -> Callable[..., Any]:
    """Select the evaluation hook and apply process isolation when requested."""
    hook = evaluate or (
        lambda config, limit=None: _run_eval_metrics(config, limit=limit, stores=stores)
    )
    if not isolate:
        return hook
    return with_isolation(
        hook,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        gpu_sampler=gpu_sampler,
    )


def storage_url(base_config: RunConfig, study_name: str, storage: str | None) -> str | None:
    """Resolve the default persistent SQLite study URL."""
    if storage is not None or not study_name:
        return storage
    db_dir = base_config.data_dir / OPTUNA_METHOD
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_dir / f'{study_name}.db'}"


def run_study(
    *,
    study_name: str,
    storage: str | None,
    goals: Sequence[str],
    seed: int,
    objective: Callable[[Any], tuple[float, ...]],
    n_trials: int,
) -> Any:
    """Create or resume an NSGA-II study and run the requested additional trials."""
    import optuna

    study = optuna.create_study(
        directions=study_directions(goals),
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials)
    return study


def study_outcome(
    study: Any,
    goals: Sequence[str],
    *,
    study_name: str,
    accuracy_floor: float | None,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Return complete/pruned trials, a robust Pareto front, and named goal picks."""
    import optuna

    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    pruned = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    if not complete:
        raise RuntimeError(f"tuning '{study_name}': no trial completed (all pruned/failed)")
    front = extract_pareto_front(list(study.best_trials) or complete, goals)
    if len(front) < 2 and len(complete) > 1:
        front = extract_pareto_front(nondominated_trials(complete, goals), goals)
    picks = select_goal_picks(
        front,
        accuracy_floor=accuracy_floor,
        include_cost=OBJECTIVE_COST in goals,
    )
    return complete, pruned, front, picks


def write_study_report(
    base_config: RunConfig,
    *,
    study_name: str,
    goals: Sequence[str],
    front: list[Any],
    picks: list[Any],
    n_trials: int,
    n_complete: int,
    n_pruned: int,
    report_dir: Path | None,
    enabled: bool,
) -> tuple[Path | None, dict[str, Path]]:
    """Write the Pareto report when enabled and return its location map."""
    if not enabled:
        return None, {}
    out_dir = report_dir or tune_run_dir(base_config.data_dir, study_name)
    paths = write_pareto_report(
        out_dir,
        study_name=study_name,
        objectives=goals,
        front=front,
        picks=picks,
        n_trials=n_trials,
        n_complete=n_complete,
        n_pruned=n_pruned,
    )
    return out_dir, paths
