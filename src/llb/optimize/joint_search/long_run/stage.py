"""The finalist phase of a confirmation run: interleaved trial blocks, then the full final split.

The stock joint-search stage tunes each finalist to completion in turn, which is fine when the
trial count is fixed in advance. A stopping rule that reads the finalist RANKING cannot work that
way -- the ranking is only meaningful between survivors that had the same budget -- so this stage
advances every finalist one block at a time and lets `sequential.run_trial_blocks` decide when to
stop.

Everything after the loop is the stock machinery: the same `score_finalist_picks` on the held-out
split, the same `result.json` resume markers, the same scoreboard writer. Only the shape of the
tuning loop changes, and the final split is scored UNCAPPED -- a confirmation run that scored a
slice of the held-out items would be the bounded acceptance run again.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.core.config import RunConfig
from llb.core.contracts.models import ResolvedModel
from llb.optimize.joint_search.hooks import candidate_config, slug
from llb.optimize.joint_search.long_run.plan import LongRunPlan
from llb.optimize.joint_search.long_run.sequential import (
    SearchTrail,
    read_trail,
    run_trial_blocks,
    write_trail,
)
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.joint_search.pick_scoring import FinalRunner, score_finalist_picks
from llb.optimize.joint_search.report import write_scoreboard
from llb.optimize.joint_search.resume import (
    read_finalist_result,
    remaining_optuna_trials,
    study_name_for,
    write_finalist_result,
)
from llb.optimize.joint_search.schedule_steps import FinalistStageRequest
from llb.optimize.joint_search.scoreboard import scoreboard_entries
from llb.optimize.tuner_models import MultiObjectiveResult

_LOG = logging.getLogger(__name__)

# The scoreboard's `recommended` row is the point-estimate argmax. In a confirmation run that is NOT
# what the run decided on, and an operator who opens `scoreboard.md` alone must not be handed a rank
# the run itself refused to act on -- so the row carries a pointer to the verdict that supersedes it.
POINT_ESTIMATE_NOTE = (
    "This row is the point-estimate argmax over the held-out board. The adoption decision for this "
    "run is the confidence-aware adopt-or-retain verdict in `long_run.md`, which can differ."
)


@dataclass
class FinalistCell:
    """One survivor's tuning context, carried across every block it is advanced through."""

    name: str
    config: RunConfig
    cell_dir: Path
    study_name: str
    tune: MultiObjectiveResult | None = None


@dataclass
class LongRunStage:
    """A `FinalistStage` that spends trials in blocks under a declared stopping rule."""

    plan: LongRunPlan
    objectives: Sequence[str]
    seed: int = 13
    isolate: bool = True
    vram_reader: Callable[[], int] | None = None
    pid_usage_reader: Callable[[], dict[int, int]] | None = None
    vram_mib: int = 0
    ram_mib: int = 0
    max_model_len: int = 8192
    # CI seams: the block tuner and the held-out runner, injected so the whole schedule is
    # exercised without an Optuna study or a backend.
    tune_block: Callable[["FinalistCell", int], MultiObjectiveResult] | None = None
    final_runner: FinalRunner | None = None
    trail: SearchTrail | None = field(default=None, init=False)

    def __call__(self, request: FinalistStageRequest) -> list[FinalistTuneResult]:
        done: list[FinalistTuneResult] = []
        cells: list[FinalistCell] = []
        for name in request.finalists:
            prior = read_finalist_result(self._cell_dir(request, name))
            if prior is None:
                cells.append(self._cell(request, name))
                continue
            _LOG.info(
                "[joint-search] long-run resume skip %s (study=%s)", prior.name, prior.study_name
            )
            done.append(prior)
        by_name = {cell.name: cell for cell in cells}
        self.trail = self._trail(request, cells, by_name)
        return self._score(request, cells, done)

    def _trail(
        self,
        request: FinalistStageRequest,
        cells: Sequence["FinalistCell"],
        by_name: dict[str, "FinalistCell"],
    ) -> SearchTrail:
        """Spend the blocks, or reload the trail a killed earlier entry already spent.

        Every finalist resuming from a finished `result.json` means there are no blocks left to
        run, and re-deriving the trail is impossible: the ranking it recorded came from tuning-split
        values this entry never computes. So the persisted trail IS the record, and a re-entry that
        finds none reports an empty one rather than inventing a stopping rule it did not apply.
        """
        if cells:
            trail = run_trial_blocks(
                [cell.name for cell in cells],
                plan=self.plan,
                advance=lambda name, target: self._advance(by_name[name], target),
            )
            write_trail(request.run_dir, trail)
            return trail
        prior = read_trail(request.run_dir)
        if prior is None:
            return run_trial_blocks([], plan=self.plan, advance=lambda _n, _t: (0.0, 0))
        _LOG.info(
            "[joint-search] long-run trail resumed: stopped by %s after %d trials per finalist",
            prior.stopped_by,
            prior.trials_per_finalist,
        )
        return prior

    def _cell_dir(self, request: FinalistStageRequest, name: str) -> Path:
        return request.run_dir / "finalists" / slug(name)

    def _cell(self, request: FinalistStageRequest, name: str) -> FinalistCell:
        resolution: ResolvedModel = request.by_name[name]
        cell_dir = self._cell_dir(request, name)
        cell_dir.mkdir(parents=True, exist_ok=True)
        return FinalistCell(
            name=name,
            config=candidate_config(
                request.base,
                resolution,
                max_model_len=self.max_model_len,
                run_name=f"joint-tune-{slug(name)}",
            ),
            cell_dir=cell_dir,
            study_name=study_name_for(request.run_dir.name, name),
        )

    def _advance(self, cell: FinalistCell, target: int) -> tuple[float, int]:
        """Top this finalist's study up to `target` trials and report its best tuning objective."""
        cell.tune = (self.tune_block or self._tune_block)(cell, target)
        best = max((point.quality for point in cell.tune.front), default=0.0)
        return best, cell.tune.n_trials

    def _tune_block(self, cell: FinalistCell, target: int) -> MultiObjectiveResult:
        """Top the study up, with every TUNING-split evaluation capped at the derived screen size.

        The derived size is what the declared power priced, so it bounds the whole tuning side of
        the run -- the halving screen and each trial alike -- and nothing else caps it. The
        held-out split is scored separately and uncapped.
        """
        from llb.optimize.multi_objective_study import tune_multi
        from llb.optimize.objectives import TrialMetrics
        from llb.optimize.tuner_runtime import _run_eval_metrics

        screen_n = self.plan.screen.applied_n
        remaining = remaining_optuna_trials(cell.config.data_dir, cell.study_name, target)
        _LOG.info(
            "[joint-search] long-run tune %s -> %d trials (%d new, %d tuning cases each)",
            cell.name,
            target,
            remaining,
            screen_n,
        )

        def evaluate(config: RunConfig, limit: int | None = None) -> TrialMetrics:
            return _run_eval_metrics(
                config, limit=screen_n if limit is None else min(limit, screen_n)
            )

        return tune_multi(
            cell.config,
            n_trials=remaining,
            study_name=cell.study_name,
            objectives=self.objectives,
            evaluate=evaluate,
            seed=self.seed,
            isolate=self.isolate,
            vram_reader=self.vram_reader,
            pid_usage_reader=self.pid_usage_reader,
            vram_mib=self.vram_mib,
            ram_mib=self.ram_mib,
            report_dir=cell.cell_dir,
            write_report=True,
            embedders=None,
            prune_case_count=screen_n,
        )

    def _score(
        self,
        request: FinalistStageRequest,
        cells: Sequence[FinalistCell],
        resumed: Sequence[FinalistTuneResult],
    ) -> list[FinalistTuneResult]:
        """Score every finalist's picks on the FULL held-out split and refresh the scoreboard."""
        results: list[FinalistTuneResult] = list(resumed)
        for cell in cells:
            if cell.tune is None:
                continue
            resolution = request.by_name[cell.name]
            finals = score_finalist_picks(
                cell.tune,
                cell.config,
                cell.cell_dir,
                final_runner=self.final_runner,
                case_limit=None,
            )
            result = FinalistTuneResult(
                name=cell.name,
                backend=resolution["chosen_backend"] or cell.config.backend,
                source=resolution["chosen_source"] or cell.config.model,
                study_name=cell.study_name,
                overrides_by_pick={
                    pick.goal: dict(pick.point.overrides) for pick in cell.tune.picks
                },
                finals=dict(finals),
                report_dir=cell.cell_dir,
            )
            write_finalist_result(cell.cell_dir, result)
            results.append(result)
            entries, recommended = scoreboard_entries(results)
            write_scoreboard(
                request.run_dir,
                run_id=request.run_dir.name,
                entries=entries,
                recommended=recommended,
                note=POINT_ESTIMATE_NOTE,
            )
        return results


def stage_summary(stage: LongRunStage) -> dict[str, Any]:
    """The trail this stage recorded, or an empty one when it never ran a block."""
    return stage.trail.to_dict() if stage.trail is not None else {}


__all__ = ["FinalistCell", "LongRunStage", "stage_summary"]
