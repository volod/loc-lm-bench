"""Multi-objective Optuna tuning: NSGA-II + MedianPruner + Pareto report."""

from pathlib import Path
from typing import Any, Callable, Sequence

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.core.contracts.runs import EvalResult
from llb.optimize.objectives import (
    OBJECTIVE_COST,
    TrialMetrics,
    extract_pareto_front,
    metrics_vector,
    nondominated_trials,
    normalize_outcome,
    parse_objectives,
    select_goal_picks,
    study_directions,
)
from llb.optimize.pareto_report import tune_run_dir, write_pareto_report
from llb.optimize.tuner_models import MultiObjectiveResult, MultiTwoStageResult
from llb.optimize.store_registry import StoreRegistry, study_stores_dir
from llb.optimize.tuner_runtime import (
    TrialCallback,
    _LOG,
    _run_eval_final,
    _run_eval_metrics,
)
from llb.optimize.tuning_space import (
    FINAL_SPLIT,
    estimate_prompt_tokens,
    fits_context,
    is_oom,
    suggest_overrides,
    with_isolation,
)

OPTUNA_METHOD = "optuna"

# Progressive case-subset fractions for median-style early pruning (before the full eval).
# Optuna's Trial.report / MedianPruner are single-objective only, so MOO implements the same
# idea by comparing subset quality to the median of prior trials' recorded step qualities.
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
    """Build an NSGA-II objective: sample -> validate -> optional subset prune -> evaluate."""
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
        return metrics_vector(metrics, objectives)

    return objective


def _evaluate_with_pruning(
    trial: Any,
    evaluate: Callable[..., Any],
    config: RunConfig,
    *,
    prune_case_count: int | None,
) -> TrialMetrics:
    """Median-style early stop on progressive case subsets, then run the full evaluate."""
    import optuna

    if prune_case_count and prune_case_count > 1:
        for step, frac in enumerate(PRUNE_FRACTIONS):
            limit = max(1, int(prune_case_count * frac))
            partial = _call_evaluate(evaluate, config, limit=limit)
            trial.set_user_attr(f"{PRUNE_ATTR_PREFIX}{step}", partial.quality)
            if _below_step_median(trial, step, partial.quality):
                raise optuna.TrialPruned(
                    f"median-pruned at subset limit={limit} quality={partial.quality:.4f}"
                )
    return _call_evaluate(evaluate, config, limit=None)


def _below_step_median(trial: Any, step: int, quality: float) -> bool:
    """True when enough prior trials exist and this subset quality is below their median."""
    import statistics

    key = f"{PRUNE_ATTR_PREFIX}{step}"
    prior = [
        float(other.user_attrs[key])
        for other in trial.study.trials
        if other.number != trial.number and key in other.user_attrs
    ]
    if len(prior) < PRUNE_WARMUP_TRIALS:
        return False
    return quality < statistics.median(prior)


def _call_evaluate(
    evaluate: Callable[..., Any], config: RunConfig, *, limit: int | None
) -> TrialMetrics:
    try:
        outcome = evaluate(config, limit=limit)
    except TypeError:
        outcome = evaluate(config)
    return normalize_outcome(outcome)


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
    """Stage 1 multi-objective search: NSGA-II over quality/latency[/cost], emit Pareto picks.

    When ``embedders`` is set and ``prewarm_stores`` is true, bake-off shortlist stores for the
    base config's chunking fingerprint are built (or loaded from
    ``$DATA_DIR/optuna/<study>/stores/``) before the Optuna loop so embedder-knob trials swap a
    cached store instead of re-embedding.
    """
    import optuna

    goals = parse_objectives(objectives)
    if OBJECTIVE_COST in goals and base_config.scorer_policy != "frontier":
        _LOG.warning(
            "[tune] cost objective requested but scorer_policy=%s; cost stays 0 unless the "
            "evaluate hook supplies it",
            base_config.scorer_policy,
        )
    shared_stores = stores is not None
    if stores is None:
        cache_dir = None
        if study_name:
            cache_dir = study_stores_dir(base_config.data_dir, study_name)
            cache_dir.mkdir(parents=True, exist_ok=True)
        stores = StoreRegistry(cache_dir=cache_dir, embedders=embedders)
    elif embedders and stores.embedders is None:
        stores.embedders = list(embedders)
    # Prewarm only when this registry feeds evaluate (default hook or caller-shared registry).
    if prewarm_stores and embedders and (evaluate is None or shared_stores):
        stores.prewarm(base_config, embedders)
    evaluate = evaluate or (
        lambda config, limit=None: _run_eval_metrics(config, limit=limit, stores=stores)
    )
    if isolate:
        evaluate = with_isolation(
            evaluate,
            vram_reader=vram_reader,
            pid_usage_reader=pid_usage_reader,
            gpu_sampler=gpu_sampler,
        )
    if storage is None and study_name:
        db_dir = base_config.data_dir / OPTUNA_METHOD
        db_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_dir / f'{study_name}.db'}"

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(
        directions=study_directions(goals),
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
        sampler=sampler,
    )
    study.optimize(
        make_multi_objective(
            base_config,
            evaluate,
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
        ),
        n_trials=n_trials,
    )

    states = optuna.trial.TrialState
    complete = [t for t in study.trials if t.state == states.COMPLETE]
    pruned = [t for t in study.trials if t.state == states.PRUNED]
    if not complete:
        raise RuntimeError(f"tuning '{study_name}': no trial completed (all pruned/failed)")
    # Prefer Optuna's non-dominated set; fall back to all complete trials.
    best_trials = list(study.best_trials) or complete
    front = extract_pareto_front(best_trials, goals)
    # If the sampler collapsed to one point, still surface any additional non-dominated
    # complete trials (epsilon-free dominance over the full complete set).
    if len(front) < 2 and len(complete) > 1:
        front = extract_pareto_front(nondominated_trials(complete, goals), goals)
    picks = select_goal_picks(
        front,
        accuracy_floor=accuracy_floor,
        include_cost=OBJECTIVE_COST in goals,
    )
    out_dir = None
    paths: dict[str, Path] = {}
    if write_report:
        out_dir = report_dir or tune_run_dir(base_config.data_dir, study_name)
        paths = write_pareto_report(
            out_dir,
            study_name=study_name,
            objectives=goals,
            front=front,
            picks=picks,
            n_trials=len(study.trials),
            n_complete=len(complete),
            n_pruned=len(pruned),
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
