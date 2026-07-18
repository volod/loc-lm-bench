"""Screening and finalist execution steps for joint search."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.models import ResolvedModel
from llb.optimize.joint_search.halving import (
    HalvingLedger,
    HalvingRound,
    ScreenScore,
    build_halving_round,
    finalize_ledger,
    screen_limit_for_round,
)
from llb.optimize.joint_search.hooks import ScreenEvaluate, candidate_config, slug
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.joint_search.report import write_ledger, write_scoreboard
from llb.optimize.joint_search.resume import (
    read_finalist_result,
    read_screen_marker,
    write_finalist_result,
    write_screen_marker,
)
from llb.optimize.joint_search.scoreboard import scoreboard_entries
from llb.optimize.tuning_space import TUNING_SPLIT

_LOG = logging.getLogger(__name__)

FinalistTune = Callable[[RunConfig, ResolvedModel, Path], FinalistTuneResult]


def partition_resolved(
    resolved: Sequence[ResolvedModel],
) -> tuple[list[ResolvedModel], list[dict[str, str]]]:
    """Separate runnable resolutions from candidates with no usable backend/source."""
    runnable: list[ResolvedModel] = []
    skipped: list[dict[str, str]] = []
    for row in resolved:
        if row["chosen_backend"] and row["chosen_source"]:
            runnable.append(row)
        else:
            skipped.append({"name": row["name"], "reason": row.get("note") or "not resolvable"})
    return runnable, skipped


def run_halving_screen(
    base: RunConfig,
    runnable: Sequence[ResolvedModel],
    *,
    run_dir: Path,
    evaluate: ScreenEvaluate,
    screen_limit: int,
    min_finalists: int,
    eta: int,
    max_model_len: int,
) -> HalvingLedger:
    """Run resumable successive-halving rounds over the tuning split."""
    active = {row["name"]: row for row in runnable}
    rounds: list[HalvingRound] = []
    round_index = 0
    while True:
        case_limit = screen_limit_for_round(screen_limit, round_index, eta=eta)
        scores = [
            _screen_candidate(
                base,
                resolution,
                run_dir=run_dir,
                evaluate=evaluate,
                round_index=round_index,
                case_limit=case_limit,
                max_model_len=max_model_len,
            )
            for _name, resolution in sorted(active.items())
        ]
        round_record = build_halving_round(
            scores,
            round_index=round_index,
            case_limit=case_limit,
            eta=eta,
            min_keep=min_finalists,
        )
        rounds.append(round_record)
        write_ledger(run_dir, finalize_ledger(rounds, eta=eta, min_finalists=min_finalists))
        if not round_record.eliminated or len(round_record.kept) <= min_finalists:
            break
        active = {name: active[name] for name in round_record.kept}
        round_index += 1
    return finalize_ledger(rounds, eta=eta, min_finalists=min_finalists)


def _screen_candidate(
    base: RunConfig,
    resolution: ResolvedModel,
    *,
    run_dir: Path,
    evaluate: ScreenEvaluate,
    round_index: int,
    case_limit: int,
    max_model_len: int,
) -> ScreenScore:
    name = resolution["name"]
    prior = read_screen_marker(run_dir, name, round_index)
    if prior is not None:
        _LOG.info("[joint-search] screen resume skip round=%d model=%s", round_index, name)
        return prior
    config = candidate_config(
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
    metrics = evaluate(config, case_limit)
    score = ScreenScore(
        name=name,
        quality=metrics.quality,
        latency_s=metrics.latency_s,
        backend=resolution["chosen_backend"] or "",
        source=resolution["chosen_source"] or "",
    )
    write_screen_marker(run_dir, score, round_index=round_index, case_limit=case_limit)
    return score


def tune_finalists(
    base: RunConfig,
    finalist_names: Sequence[str],
    by_name: dict[str, ResolvedModel],
    *,
    run_dir: Path,
    tuner: FinalistTune,
) -> list[FinalistTuneResult]:
    """Tune or resume finalists and refresh the partial scoreboard after each one."""
    results: list[FinalistTuneResult] = []
    for name in finalist_names:
        resolution = by_name[name]
        cell_dir = run_dir / "finalists" / slug(name)
        cell_dir.mkdir(parents=True, exist_ok=True)
        prior = read_finalist_result(cell_dir)
        if prior is not None:
            _LOG.info("[joint-search] finalist resume skip %s (study=%s)", name, prior.study_name)
            results.append(prior)
        else:
            _LOG.info(
                "[joint-search] deep-tuning finalist %s (%s)",
                name,
                resolution["chosen_backend"],
            )
            result = tuner(base, resolution, cell_dir)
            write_finalist_result(cell_dir, result)
            results.append(result)
        entries, recommended = scoreboard_entries(results)
        write_scoreboard(run_dir, run_id=run_dir.name, entries=entries, recommended=recommended)
    return results
