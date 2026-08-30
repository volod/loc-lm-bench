"""Spend multi-objective trials in blocks until the ranking settles or the budget runs out.

The loop is pure with respect to tuning: it advances every finalist to the same cumulative trial
count, asks the injected `advance` hook what that finalist's best TUNING-split objective now is,
and hands the block to the stability rule. Because it never sees a final-split number, the held-out
score cannot influence how long the search ran.

The trail it returns is the audit record the acceptance gate asks for: which rule stopped the
search, and what the search actually cost.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.optimize.joint_search.long_run.plan import LongRunPlan
from llb.optimize.joint_search.long_run.stability import BlockSnapshot, build_snapshot

_LOG = logging.getLogger(__name__)

STOPPED_BY_STABILITY = "ranking-stability"
STOPPED_BY_BUDGET = "trial-budget"

# The trail is the ONLY part of the record no other artifact holds: the finalist `result.json`
# files carry the tuned picks and their held-out scores, and a resumed run reloads those, but the
# block ranking that stopped the search exists nowhere else. Without persisting it, re-entering a
# killed run -- the exact recovery the resume markers exist for -- rewrites `long_run.json` with an
# empty trail and the artifact can no longer state which rule ended the search.
TRAIL_FILENAME = "search_trail.json"

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


def trail_path(run_dir: Path) -> Path:
    """Where a confirmation run keeps its block trail across a kill and a re-entry."""
    return run_dir / TRAIL_FILENAME


def write_trail(run_dir: Path, trail: SearchTrail) -> Path:
    """Persist the trail beside the run's other resume markers."""
    path = trail_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trail.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_trail(run_dir: Path) -> SearchTrail | None:
    """The trail an earlier entry of this run recorded, or None when there is none to reuse."""
    path = trail_path(run_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[joint-search] ignore unreadable search trail %s: %s", path, exc)
        return None
    try:
        blocks = tuple(_snapshot_from(block) for block in payload["blocks"])
        return SearchTrail(
            blocks=blocks,
            stopped_by=str(payload["stopped_by"]),
            trials_per_finalist=int(payload["trials_per_finalist"]),
            consumed_trials={str(k): int(v) for k, v in payload["consumed_trials"].items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.warning("[joint-search] ignore malformed search trail %s: %s", path, exc)
        return None


def _snapshot_from(block: dict[str, Any]) -> BlockSnapshot:
    """Rebuild one persisted block; the snapshot is data, so nothing is recomputed here."""
    return BlockSnapshot(
        index=int(block["index"]),
        trials_per_finalist=int(block["trials_per_finalist"]),
        consumed_trials=int(block["consumed_trials"]),
        ranking=tuple(block["ranking"]),
        objective={str(k): float(v) for k, v in block["objective"].items()},
        agreement=None if block["agreement"] is None else float(block["agreement"]),
        leader_held=None if block["leader_held"] is None else bool(block["leader_held"]),
        stable=bool(block["stable"]),
        stable_streak=int(block["stable_streak"]),
    )


__all__ = [
    "STOPPED_BY_BUDGET",
    "STOPPED_BY_STABILITY",
    "TRAIL_FILENAME",
    "AdvanceFinalist",
    "SearchTrail",
    "read_trail",
    "run_trial_blocks",
    "trail_path",
    "write_trail",
]
