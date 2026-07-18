"""Result dataclasses for single- and multi-objective Optuna tuning."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
from llb.optimize.objectives import GoalPick, ParetoPoint


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


@dataclass
class MultiObjectiveResult:
    """Stage-1 multi-objective study: Pareto front plus named per-goal picks."""

    study_name: str
    storage: str | None
    objectives: tuple[str, ...]
    n_trials: int
    n_complete: int
    n_pruned: int
    front: list[ParetoPoint]
    picks: list[GoalPick]
    report_dir: Path | None = None
    report_paths: dict[str, Path] = field(default_factory=dict)
    store_builds: list[tuple[Any, ...]] = field(default_factory=list)

    def config_for(self, base: RunConfig, pick_goal: str) -> RunConfig:
        """Apply the overrides of a named pick onto ``base``."""
        for pick in self.picks:
            if pick.goal == pick_goal:
                return base.with_overrides(**pick.point.overrides)
        raise KeyError(f"no pick named {pick_goal!r}; have {[p.goal for p in self.picks]}")


@dataclass
class MultiTwoStageResult:
    """Stage-1 Pareto search plus stage-2 final-split scores for each named pick."""

    tune: MultiObjectiveResult
    finals: dict[str, EvalResult]
