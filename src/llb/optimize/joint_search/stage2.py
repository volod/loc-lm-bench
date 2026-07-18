"""Stage-2 final-split pick scoring with per-pick resume markers."""

import logging
from pathlib import Path
from typing import Callable

from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
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
) -> dict[str, EvalResult]:
    """Score each named pick on the final split; reuse ``picks/<goal>.json`` when present.

    A kill mid-stage-2 leaves completed pick markers in place so a resume skips those evals
    even when the Optuna study is already full and ``result.json`` was never written.
    """
    from llb.optimize.tuner_runtime import _run_eval_final

    runner = final_runner or _run_eval_final
    finals: dict[str, EvalResult] = {}
    for pick in tune.picks:
        prior = read_pick_marker(cell_dir, pick.goal)
        if prior is not None:
            _LOG.info("[joint-search] stage-2 resume skip pick=%s", pick.goal)
            finals[pick.goal] = prior
            continue
        cfg = tune.config_for(base, pick.goal)
        _LOG.info(
            "[joint-search] stage-2 scoring pick=%s on the '%s' split",
            pick.goal,
            FINAL_SPLIT,
        )
        outcome = runner(cfg)
        write_pick_marker(cell_dir, pick.goal, outcome)
        finals[pick.goal] = outcome
    return finals
