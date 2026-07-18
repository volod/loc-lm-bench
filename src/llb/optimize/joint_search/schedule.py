"""Joint model + RAG-config search: screen -> successive-halving -> per-finalist tune."""

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec, ResolvedModel
from llb.optimize.joint_search.constants import (
    DEFAULT_ETA,
    DEFAULT_MIN_FINALISTS,
    DEFAULT_OBJECTIVES,
    DEFAULT_SCREEN_LIMIT,
)
from llb.optimize.joint_search.halving import (
    HalvingLedger,
    HalvingRound,
    ScreenScore,
    build_halving_round,
    finalize_ledger,
    screen_limit_for_round,
)
from llb.optimize.joint_search.hooks import (
    ScreenEvaluate,
    candidate_config,
    default_screen_evaluate,
    default_tune_finalist,
    slug,
    wrap_screen_isolation,
)
from llb.optimize.joint_search.models import FinalistTuneResult, JointSearchResult
from llb.optimize.joint_search.report import (
    joint_run_dir,
    write_ledger,
    write_manifest,
    write_scoreboard,
)
from llb.optimize.joint_search.scoreboard import scoreboard_entries
from llb.optimize.objectives import parse_objectives
from llb.optimize.tuning_space import FINAL_SPLIT, TUNING_SPLIT

_LOG = logging.getLogger(__name__)

FinalistTune = Callable[[RunConfig, ResolvedModel, Path], FinalistTuneResult]


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
    """
    from llb.backends.resolver import resolve_all

    goals = parse_objectives(objectives)
    run_dir = joint_run_dir(base_config.data_dir, run_id)
    resolved = resolve_all(list(candidates), vram_mib, ram_mib, probes=probes)
    runnable, skipped = _partition_resolved(resolved)
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

    ledger = _run_halving_screen(
        base_config,
        runnable,
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
    finalist_results: list[FinalistTuneResult] = []
    for name in ledger.finalists:
        resolution = by_name[name]
        _LOG.info("[joint-search] deep-tuning finalist %s (%s)", name, resolution["chosen_backend"])
        cell_dir = run_dir / "finalists" / slug(name)
        cell_dir.mkdir(parents=True, exist_ok=True)
        finalist_results.append(tuner(base_config, resolution, cell_dir))

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


def _partition_resolved(
    resolved: Sequence[ResolvedModel],
) -> tuple[list[ResolvedModel], list[dict[str, str]]]:
    runnable: list[ResolvedModel] = []
    skipped: list[dict[str, str]] = []
    for row in resolved:
        if row["chosen_backend"] and row["chosen_source"]:
            runnable.append(row)
        else:
            skipped.append({"name": row["name"], "reason": row.get("note") or "not resolvable"})
    return runnable, skipped


def _run_halving_screen(
    base: RunConfig,
    runnable: Sequence[ResolvedModel],
    *,
    evaluate: ScreenEvaluate,
    screen_limit: int,
    min_finalists: int,
    eta: int,
    max_model_len: int,
) -> HalvingLedger:
    active = {r["name"]: r for r in runnable}
    rounds: list[HalvingRound] = []
    round_index = 0
    while True:
        case_limit = screen_limit_for_round(screen_limit, round_index, eta=eta)
        scores: list[ScreenScore] = []
        for name, resolution in sorted(active.items()):
            cfg = candidate_config(
                base,
                resolution,
                max_model_len=max_model_len,
                run_name=f"joint-screen-{slug(name)}-r{round_index}",
            )
            _LOG.info(
                "[joint-search] screen round=%d model=%s limit=%d split=%s",
                round_index,
                name,
                case_limit,
                TUNING_SPLIT,
            )
            metrics = evaluate(cfg, case_limit)
            scores.append(
                ScreenScore(
                    name=name,
                    quality=metrics.quality,
                    latency_s=metrics.latency_s,
                    backend=resolution["chosen_backend"] or "",
                    source=resolution["chosen_source"] or "",
                )
            )
        round_rec = build_halving_round(
            scores,
            round_index=round_index,
            case_limit=case_limit,
            eta=eta,
            min_keep=min_finalists,
        )
        rounds.append(round_rec)
        if not round_rec.eliminated or len(round_rec.kept) <= min_finalists:
            break
        active = {name: active[name] for name in round_rec.kept}
        round_index += 1
    return finalize_ledger(rounds, eta=eta, min_finalists=min_finalists)
