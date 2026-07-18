"""CLI helper for multi-objective `llb tune --objectives`."""

from typing import Any

import typer

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.optimize.objectives import TrialMetrics


def run_multi_objective_tune(
    cfg: RunConfig,
    *,
    objectives: str,
    trials: int,
    study_name: str,
    model: str,
    backend: str,
    spec: ModelSpec | None,
    gpus: list[Any],
    seed: int,
    isolate: bool,
    vram_reader: Any,
    pid_reader: Any,
    strategies: list[str] | None,
    tune_reranker: str | None,
    embedders: str | None,
    context_budget: bool,
    accuracy_floor: float | None,
    limit: int | None = None,
) -> None:
    """Execute NSGA-II tune + stage-2 finals and print the Pareto / pick summary."""
    from llb.backends.hardware import detect_ram_mb, max_vram_mb
    from llb.optimize.multi_objective import two_stage_multi
    from llb.optimize.objectives import OBJECTIVE_COST, parse_objectives
    from llb.optimize.tuner_runtime import StoreRegistry, _run_eval_metrics

    goals = parse_objectives(objectives)
    if OBJECTIVE_COST in goals and cfg.scorer_policy != "frontier":
        typer.echo(
            "[tune] cost objective requires scorer_policy=frontier (consent + budget); "
            "refusing so spend is never silent",
            err=True,
        )
        raise typer.Exit(code=2)
    embedder_list = _parse_embedders(embedders)
    stores = StoreRegistry()
    case_limit = limit

    def evaluate(config: RunConfig, limit: int | None = None) -> TrialMetrics:
        # CLI case_limit caps the full trial; MedianPruner subsets pass a smaller limit.
        if limit is None:
            capped = case_limit
        elif case_limit is None:
            capped = limit
        else:
            capped = min(limit, case_limit)
        return _run_eval_metrics(config, limit=capped, stores=stores)

    typer.echo(
        f"[tune] study={study_name} model={model} backend={backend} trials={trials} "
        f"objectives={','.join(goals)}"
    )
    out = two_stage_multi(
        cfg,
        n_trials=trials,
        study_name=study_name,
        objectives=goals,
        evaluate=evaluate,
        model_spec=spec,
        vram_mib=max_vram_mb(gpus),
        ram_mib=detect_ram_mb(),
        seed=seed,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        strategies=strategies,
        reranker=tune_reranker,
        embedders=embedder_list,
        tune_context_budget=context_budget,
        prune_case_count=limit,
        accuracy_floor=accuracy_floor,
    )
    t = out.tune
    t.store_builds = list(stores.builds)
    typer.echo(
        f"[tune] stage-1 Pareto front={len(t.front)} "
        f"({t.n_complete} complete, {t.n_pruned} pruned of {t.n_trials})"
    )
    for pick in t.picks:
        point = pick.point
        typer.echo(
            f"[tune] pick {pick.goal}: trial={point.number} quality={point.quality:.4f} "
            f"latency_s={point.latency_s:.3f} cost_usd={point.cost_usd:.4f} "
            f"overrides={point.overrides}"
        )
    if t.report_paths:
        report = t.report_paths.get("markdown")
        typer.echo(f"[tune] Pareto report: {report}")
    typer.echo("[tune] stage-2 (final split) per pick:")
    for goal, final in out.finals.items():
        typer.echo(f"[tune] final[{goal}]:")
        typer.echo(final["table"])


def _parse_embedders(embedders: str | None) -> list[str] | None:
    """None -> bake-off shortlist; empty string -> pinned; else comma list."""
    from llb.rag.embedding_bakeoff import DEFAULT_LOCAL_CANDIDATES

    if embedders is None:
        return list(DEFAULT_LOCAL_CANDIDATES)
    if embedders.strip() == "":
        return None
    parts = [part.strip() for part in embedders.split(",") if part.strip()]
    return parts or None
