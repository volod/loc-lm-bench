"""Orchestrate a research-scale roster confirmation on top of the ordinary joint search.

The bounded acceptance run and this one execute the SAME schedule -- host-fit filter, cheap
tuning-split screen, successive halving, per-finalist multi-objective tune, final-split scoreboard.
What a confirmation adds is everything that turns a scoreboard into a decision, and every piece of
it is bolted on at a declared seam rather than forked into a second pipeline:

1. the screen case cap comes from `plan.screen`, derived from paired power instead of chosen;
2. the finalist phase is the interleaved `LongRunStage`, so the stopping rule can read a ranking;
3. the final split is scored UNCAPPED;
4. the finalists are screened on the public Ukrainian tracks;
5. the board is re-read per case for intervals, paired deltas, and the quality/latency frontier;
6. the verdict is stated as adopt-or-retain against a declared incumbent.

Steps 4-6 never touch the tuning loop, so no held-out number can reach the search.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.optimize.joint_search.long_run.plan import LONG_RUN_METHOD, LongRunPlan
from llb.optimize.joint_search.long_run.public_tracks import (
    SCREEN_METHOD,
    ScreenRunner,
    default_screen_runner,
    screen_finalists,
)
from llb.optimize.joint_search.long_run.report import build_payload, write_long_run
from llb.optimize.joint_search.long_run.stage import (
    POINT_ESTIMATE_NOTE,
    LongRunStage,
    stage_summary,
)
from llb.optimize.joint_search.long_run.uncertainty import (
    BoardRow,
    BoardUncertainty,
    paired_deltas,
    read_board_rows,
    read_uncertainty,
    strongest_challenger,
)
from llb.optimize.joint_search.long_run.verdict import AdoptionVerdict, decide
from llb.optimize.joint_search.hooks import ScreenEvaluate
from llb.optimize.joint_search.models import JointSearchResult
from llb.optimize.joint_search.schedule import run_joint_search
from llb.optimize.objectives import parse_objectives
from llb.rag.fusion_evidence.power import PowerAnalysis, resolve_power_analysis
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LongRunResult:
    """The joint-search result plus everything the adoption decision rests on."""

    search: JointSearchResult
    plan: LongRunPlan
    trail: dict[str, Any]
    uncertainty: BoardUncertainty
    public: dict[str, Any]
    verdict: AdoptionVerdict
    paths: dict[str, Path]


def run_long_run(
    base_config: RunConfig,
    candidates: Sequence[ModelSpec],
    *,
    plan: LongRunPlan,
    incumbent: str | None,
    run_id: str | None = None,
    min_finalists: int = 2,
    eta: int = 2,
    objectives: str | Sequence[str] = "quality,latency",
    vram_mib: int = 0,
    ram_mib: int = 0,
    probes: Any | None = None,
    isolate: bool = True,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    seed: int = 13,
    max_model_len: int = 8192,
    screen_runner: ScreenRunner | None = None,
    screen_evaluate: ScreenEvaluate | None = None,
    stage: LongRunStage | None = None,
    public_limit: int | None = None,
    public_evict: bool = True,
    resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = DEFAULT_SEED,
) -> LongRunResult:
    """Run the confirmation end to end and write `long_run.{json,md}` beside the scoreboard."""
    goals = parse_objectives(objectives)
    stage = stage or LongRunStage(
        plan=plan,
        objectives=goals,
        seed=seed,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        vram_mib=vram_mib,
        ram_mib=ram_mib,
        max_model_len=max_model_len,
    )
    search = run_joint_search(
        base_config,
        candidates,
        n_trials=plan.trial_budget,
        run_id=run_id,
        screen_limit=plan.screen.applied_n,
        min_finalists=min_finalists,
        eta=eta,
        objectives=goals,
        vram_mib=vram_mib,
        ram_mib=ram_mib,
        probes=probes,
        screen_evaluate=screen_evaluate,
        finalist_stage=stage,
        manifest_extra={"mode": LONG_RUN_METHOD, "incumbent": incumbent},
        scoreboard_note=POINT_ESTIMATE_NOTE,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        seed=seed,
        max_model_len=max_model_len,
        case_limit=None,
        screen_case_cap=plan.screen.applied_n,
    )
    public = screen_finalists(
        [{"name": f.name, "backend": f.backend, "source": f.source} for f in search.finalists],
        out_dir=base_config.data_dir / SCREEN_METHOD,
        runner=screen_runner
        or default_screen_runner(
            base_config.with_overrides(max_model_len=max_model_len),
            limit=public_limit,
            isolate=isolate,
            evict=public_evict,
            vram_reader=vram_reader,
            pid_usage_reader=pid_usage_reader,
        ),
        limit=public_limit,
    )
    entries = _scoreboard_entries(search)
    rows, unreadable = read_board_rows(entries, {f.name: f.finals for f in search.finalists})
    uncertainty = read_uncertainty(
        rows,
        incumbent=incumbent,
        unreadable=unreadable,
        confidence=plan.confidence,
        resamples=resamples,
        seed=bootstrap_seed,
    )
    verdict = decide(uncertainty, incumbent=incumbent, public=public)
    realized = realized_power(plan, rows, uncertainty)
    trail = stage_summary(stage)
    paths = write_long_run(
        search.run_dir,
        build_payload(
            run_id=search.run_id,
            plan=plan,
            search=trail,
            ledger=search.ledger.to_dict(),
            entries=entries,
            uncertainty=uncertainty,
            realized_power=realized,
            public=public,
            verdict=verdict,
        ),
    )
    _LOG.info(
        "[joint-search] long-run verdict=%s model=%s -> %s",
        verdict.decision,
        verdict.model,
        paths["markdown"],
    )
    return LongRunResult(
        search=search,
        plan=plan,
        trail=trail,
        uncertainty=uncertainty,
        public=public,
        verdict=verdict,
        paths=paths,
    )


def realized_power(
    plan: LongRunPlan, rows: Sequence[BoardRow], uncertainty: BoardUncertainty
) -> PowerAnalysis | None:
    """Re-price the declaration with the run's OWN variance on the held-out split.

    The declared half says what the run planned for; this says what the item set it actually
    reached can resolve, and whether the strongest challenger's delta settles at all. A quieter
    reference set cannot make an underpowered run look complete, which is the whole point of the
    shared contract's second half.
    """
    candidate = strongest_challenger(uncertainty)
    if candidate is None or uncertainty.baseline is None:
        return None
    deltas = paired_deltas(rows, candidate, uncertainty.baseline)
    if len(deltas) < 2:
        return None
    return resolve_power_analysis(
        plan.power,
        deltas,
        uncertainty.paired[candidate],
        candidate=candidate,
        baseline=uncertainty.baseline,
    )


def _scoreboard_entries(search: JointSearchResult) -> list[dict[str, Any]]:
    """Re-read the entries the run's own scoreboard writer just persisted (leak fence included)."""
    import json

    path = search.scoreboard_paths.get("json")
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    return [dict(entry) for entry in entries] if isinstance(entries, list) else []


__all__ = ["LongRunResult", "realized_power", "run_long_run"]
