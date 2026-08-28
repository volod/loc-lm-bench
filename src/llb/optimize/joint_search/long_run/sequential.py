"""Spend multi-objective trials in blocks until the ranking settles or the budget runs out.

The loop is pure with respect to tuning: it advances every finalist to the same cumulative trial
count, asks the injected `advance` hook what that finalist's best TUNING-split objective now is,
and hands the block to the stability rule. Because it never sees a final-split number, the held-out
score cannot influence how long the search ran.

The trail it returns is the audit record the acceptance gate asks for: which rule stopped the
search, and what the search actually cost.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from llb.optimize.joint_search.long_run.plan import LongRunPlan
from llb.optimize.joint_search.long_run.stability import BlockSnapshot, build_snapshot

_LOG = logging.getLogger(__name__)

STOPPED_BY_STABILITY = "ranking-stability"
STOPPED_BY_BUDGET = "trial-budget"

# (finalist, cumulative trial target) -> (best tuning-split objective, trials the study now holds).
AdvanceFinalist = Callable[[str, int], tuple[float, int]]


@dataclass(frozen=True)
class SearchTrail:
    """Every block the search spent, and the rule that ended it."""

    blocks: tuple[BlockSnapshot, ...]
    stopped_by: str
    trials_per_finalist: int
    consumed_trials: dict[str, int]

    @property
    def consumed_total(self) -> int:
        return sum(self.consumed_trials.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "stopped_by": self.stopped_by,
            "trials_per_finalist": self.trials_per_finalist,
            "consumed_trials": dict(self.consumed_trials),
            "consumed_total": self.consumed_total,
            "budget_exhausted": self.stopped_by == STOPPED_BY_BUDGET,
        }


def run_trial_blocks(
    finalists: Sequence[str],
    *,
    plan: LongRunPlan,
    advance: AdvanceFinalist,
) -> SearchTrail:
    """Advance every finalist one block at a time until the stopping rule fires."""
    if not finalists:
        return SearchTrail(
            blocks=(), stopped_by=STOPPED_BY_BUDGET, trials_per_finalist=0, consumed_trials={}
        )
    blocks: list[BlockSnapshot] = []
    consumed: dict[str, int] = {name: 0 for name in finalists}
    target = 0
    stopped_by = STOPPED_BY_BUDGET
    while target < plan.trial_budget:
        target = min(target + plan.trial_block, plan.trial_budget)
        objective: dict[str, float] = {}
        for name in finalists:
            value, held = advance(name, target)
            objective[name] = value
            consumed[name] = held
        snapshot = build_snapshot(
            index=len(blocks),
            trials_per_finalist=target,
            consumed_trials=sum(consumed.values()),
            objective=objective,
            previous=blocks[-1] if blocks else None,
            agreement_floor=plan.stability_agreement,
        )
        blocks.append(snapshot)
        _LOG.info(
            "[joint-search] long-run block=%d trials/finalist=%d ranking=%s agreement=%s streak=%d",
            snapshot.index,
            snapshot.trials_per_finalist,
            list(snapshot.ranking),
            "-" if snapshot.agreement is None else f"{snapshot.agreement:.2f}",
            snapshot.stable_streak,
        )
        if snapshot.stable_streak >= plan.stability_blocks:
            stopped_by = STOPPED_BY_STABILITY
            break
    return SearchTrail(
        blocks=tuple(blocks),
        stopped_by=stopped_by,
        trials_per_finalist=target,
        consumed_trials=consumed,
    )


__all__ = [
    "STOPPED_BY_BUDGET",
    "STOPPED_BY_STABILITY",
    "AdvanceFinalist",
    "SearchTrail",
    "run_trial_blocks",
]
