"""Final-split pick scoring with per-pick resume markers."""

import logging
from pathlib import Path
from typing import Callable

from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
from llb.executor.sweep_cells import cell_key
from llb.optimize.joint_search.resume import read_pick_marker, write_pick_marker
from llb.optimize.tuner_models import MultiObjectiveResult
from llb.optimize.tuning_space import FINAL_SPLIT

_LOG = logging.getLogger(__name__)

FinalRunner = Callable[[RunConfig], EvalResult]


def score_finalist_picks(
    tune: MultiObjectiveResult,
    base: RunConfig,
    cell_dir: Path,
    *,
    final_runner: FinalRunner | None = None,
    case_limit: int | None = None,
) -> dict[str, EvalResult]:
    """Score each named pick on the final split; reuse ``picks/<goal>.json`` when present.

    A kill mid-pick-scoring leaves completed pick markers in place so a resume skips those
    evals even when the Optuna study is already full and ``result.json`` was never written.
    """
    from llb.optimize.tuner_runtime import _run_eval_final

    if final_runner is not None:
        runner = final_runner
    elif case_limit is None:
        runner = _run_eval_final
    else:

        def runner(config: RunConfig) -> EvalResult:
            return _run_eval_final(config, limit=case_limit)

    finals: dict[str, EvalResult] = {}
    outcomes_by_cell: dict[str, EvalResult] = {}
    for pick in tune.picks:
        cfg = tune.config_for(base, pick.goal)
        key = cell_key(cfg)
        prior = read_pick_marker(cell_dir, pick.goal)
        if prior is not None:
            _LOG.info("[joint-search] pick-scoring resume skip pick=%s", pick.goal)
            finals[pick.goal] = prior
            outcomes_by_cell[key] = prior
            continue
        if key in outcomes_by_cell:
            _LOG.info("[joint-search] pick-scoring reuse identical config pick=%s", pick.goal)
            outcome = outcomes_by_cell[key]
            write_pick_marker(cell_dir, pick.goal, outcome)
            finals[pick.goal] = outcome
            continue
        _LOG.info(
            "[joint-search] pick-scoring pick=%s on the '%s' split",
            pick.goal,
            FINAL_SPLIT,
        )
        outcome = runner(cfg)
        write_pick_marker(cell_dir, pick.goal, outcome)
        finals[pick.goal] = outcome
        outcomes_by_cell[key] = outcome
    return finals
