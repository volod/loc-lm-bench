"""Default screen / tune hooks and isolation wrappers for joint-search."""

from pathlib import Path
from typing import Callable, Sequence

from llb.core.config import RunConfig
from llb.core.contracts.models import ResolvedModel
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.objectives import TrialMetrics

ScreenEvaluate = Callable[[RunConfig, int | None], TrialMetrics]


def candidate_config(
    base: RunConfig,
    resolution: ResolvedModel,
    *,
    max_model_len: int,
    run_name: str,
) -> RunConfig:
    """Build a RunConfig for one resolved candidate."""
    overrides: dict[str, object] = {
        "model": resolution["chosen_source"],
        "backend": resolution["chosen_backend"],
        "run_name": run_name,
    }
    if resolution["chosen_backend"] == "vllm":
        overrides["max_model_len"] = max_model_len
    return base.with_overrides(**overrides)


def default_screen_evaluate(config: RunConfig, limit: int | None) -> TrialMetrics:
    """Tuning-split metrics with an optional case cap (cheap screen)."""
    from llb.optimize.tuner_runtime import _run_eval_metrics

    return _run_eval_metrics(config, limit=limit)


def wrap_screen_isolation(
    evaluate: ScreenEvaluate,
    *,
    vram_reader: Callable[[], int] | None,
    pid_usage_reader: Callable[[], dict[int, int]] | None,
) -> ScreenEvaluate:
    """Run each screen cell under the process isolation reclaim contract."""
    from llb.executor.isolation import isolate_cell

    def wrapped(config: RunConfig, limit: int | None) -> TrialMetrics:
        def work() -> TrialMetrics:
            return evaluate(config, limit)

        result, _iso = isolate_cell(
            work,
            backend=config.backend,
            vram_reader=vram_reader,
            pid_usage_reader=pid_usage_reader,
        )
        return result

    return wrapped


def default_tune_finalist(
    base: RunConfig,
    resolution: ResolvedModel,
    cell_dir: Path,
    *,
    n_trials: int,
    objectives: Sequence[str],
    seed: int,
    isolate: bool,
    vram_reader: Callable[[], int] | None,
    pid_usage_reader: Callable[[], dict[int, int]] | None,
    vram_mib: int,
    ram_mib: int,
    max_model_len: int,
    case_limit: int | None = None,
) -> FinalistTuneResult:
    """Per-finalist multi-objective two-stage tune into ``cell_dir``."""
    from llb.optimize.multi_objective import two_stage_multi
    from llb.optimize.objectives import TrialMetrics
    from llb.optimize.tuner_runtime import _run_eval_metrics

    name = resolution["name"]
    cfg = candidate_config(
        base,
        resolution,
        max_model_len=max_model_len,
        run_name=f"joint-tune-{slug(name)}",
    )
    study_name = f"joint-{cell_dir.parent.parent.name}-{slug(name)}"

    def evaluate(config: RunConfig, limit: int | None = None) -> TrialMetrics:
        if limit is None:
            capped = case_limit
        elif case_limit is None:
            capped = limit
        else:
            capped = min(limit, case_limit)
        return _run_eval_metrics(config, limit=capped)

    out = two_stage_multi(
        cfg,
        n_trials=n_trials,
        study_name=study_name,
        objectives=objectives,
        evaluate=evaluate,
        seed=seed,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        vram_mib=vram_mib,
        ram_mib=ram_mib,
        report_dir=cell_dir,
        write_report=True,
        embedders=None,
        prune_case_count=case_limit,
    )
    overrides_by_pick = {pick.goal: dict(pick.point.overrides) for pick in out.tune.picks}
    return FinalistTuneResult(
        name=name,
        backend=resolution["chosen_backend"] or cfg.backend,
        source=resolution["chosen_source"] or cfg.model,
        study_name=study_name,
        overrides_by_pick=overrides_by_pick,
        finals=dict(out.finals),
        report_dir=cell_dir,
    )


def slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")
