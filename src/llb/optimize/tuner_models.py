"""Focused tuner models implementation."""

from dataclasses import dataclass
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult


@dataclass
class TuneResult:
    best_config: RunConfig
    best_value: float
    n_trials: int
    n_complete: int
    n_pruned: int
    study_name: str
    storage: str | None


@dataclass
class TwoStageResult:
    tune: TuneResult
    final: EvalResult  # the stage-2 run on the full final split -- the leaderboard entry
