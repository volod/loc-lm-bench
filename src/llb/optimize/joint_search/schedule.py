"""Joint model + RAG-config search: screen -> successive-halving -> per-finalist tune."""

from typing import Any, Callable, Sequence

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.optimize.joint_search.constants import (
    DEFAULT_ETA,
    DEFAULT_MIN_FINALISTS,
    DEFAULT_OBJECTIVES,
    DEFAULT_SCREEN_LIMIT,
)
from llb.optimize.joint_search.halving import finalize_ledger
from llb.optimize.joint_search.hooks import (
    ScreenEvaluate,
    default_screen_evaluate,
    default_tune_finalist,
    wrap_screen_isolation,
)
from llb.optimize.joint_search.models import JointSearchResult
from llb.optimize.joint_search.report import (
    joint_run_dir,
    write_ledger,
    write_manifest,
    write_scoreboard,
)
from llb.optimize.joint_search.schedule_steps import (
    FinalistTune,
    partition_resolved,
    run_halving_screen,
    tune_finalists,
)
from llb.optimize.joint_search.scoreboard import scoreboard_entries
from llb.optimize.objectives import parse_objectives
from llb.optimize.tuning_space import FINAL_SPLIT, TUNING_SPLIT


def run_joint_search(
    base_config: RunConfig,
    candidates: Sequence[ModelSpec],
    *,
    n_trials: int,
    run_id: str | None = None,
    screen_limit: int = DEFAULT_SCREEN_LIMIT,
    min_finalists: int = DEFAULT_MIN_FINALISTS,
    eta: int = DEFAULT_ETA,
    objectives: str | Sequence[str] = DEFAULT_OBJECTIVES,
    vram_mib: int = 0,
    ram_mib: int = 0,
    probes: Any | None = None,
    screen_evaluate: ScreenEvaluate | None = None,
    tune_finalist: FinalistTune | None = None,
    isolate: bool = True,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    seed: int = 13,
    max_model_len: int = 8192,
    case_limit: int | None = None,
) -> JointSearchResult:
    """Resolve -> cheap tuning-split screen with successive-halving -> deep-tune survivors.

    Screen and elimination scores always use ``TUNING_SPLIT``. The scoreboard is built
    exclusively from final-split pick scores (leak fence enforced in writers).

    Re-entry with the same ``run_id`` resumes: completed screen markers are skipped, finished
    finalist ``result.json`` files are reloaded, and Optuna studies under ``$DATA_DIR/optuna/``
    only run remaining trials.
    """
    from llb.backends.resolver import resolve_all

    goals = parse_objectives(objectives)
    run_dir = joint_run_dir(base_config.data_dir, run_id)
    resolved = resolve_all(list(candidates), vram_mib, ram_mib, probes=probes)
    runnable, skipped = partition_resolved(resolved, data_dir=base_config.data_dir)
    write_manifest(
        run_dir,
        {
            "run_id": run_dir.name,
            "candidates": [c["name"] for c in candidates],
            "runnable": [r["name"] for r in runnable],
            "skipped": skipped,
            "screen_limit": screen_limit,
            "min_finalists": min_finalists,
            "eta": eta,
            "n_trials": n_trials,
            "objectives": list(goals),
            "screen_split": TUNING_SPLIT,
            "scoreboard_split": FINAL_SPLIT,
            "seed": seed,
            "case_limit": case_limit,
        },
    )
    if not runnable:
        empty = finalize_ledger([], eta=eta, min_finalists=min_finalists)
        write_ledger(run_dir, empty)
        paths = write_scoreboard(run_dir, run_id=run_dir.name, entries=[], recommended=None)
        return JointSearchResult(
            run_id=run_dir.name,
            run_dir=run_dir,
            ledger=empty,
            finalists=[],
            scoreboard_paths=paths,
            recommended=None,
            skipped=skipped,
        )

    evaluate = screen_evaluate or default_screen_evaluate
    if isolate and screen_evaluate is None:
        evaluate = wrap_screen_isolation(
            evaluate, vram_reader=vram_reader, pid_usage_reader=pid_usage_reader
        )

    ledger = run_halving_screen(
        base_config,
        runnable,
        run_dir=run_dir,
        evaluate=evaluate,
        screen_limit=screen_limit,
        min_finalists=min_finalists,
        eta=eta,
        max_model_len=max_model_len,
    )
    write_ledger(run_dir, ledger)
    by_name = {r["name"]: r for r in runnable}
    tuner = tune_finalist or (
        lambda cfg, resolution, out: default_tune_finalist(
            cfg,
            resolution,
            out,
            n_trials=n_trials,
            objectives=goals,
            seed=seed,
            isolate=isolate,
            vram_reader=vram_reader,
            pid_usage_reader=pid_usage_reader,
            vram_mib=vram_mib,
            ram_mib=ram_mib,
            max_model_len=max_model_len,
            case_limit=case_limit,
        )
    )
    finalist_results = tune_finalists(
        base_config, ledger.finalists, by_name, run_dir=run_dir, tuner=tuner
    )
    entries, recommended = scoreboard_entries(finalist_results)
    paths = write_scoreboard(run_dir, run_id=run_dir.name, entries=entries, recommended=recommended)
    return JointSearchResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        ledger=ledger,
        finalists=finalist_results,
        scoreboard_paths=paths,
        recommended=recommended,
        skipped=skipped,
    )
