"""When has searching harder stopped changing the answer? (pure; no I/O)

A fixed trial count is the wrong stopping rule for a confirmation run: too few and the ranking is
still moving, too many and the GPU hours buy nothing. The rule here reads the only thing the
decision rests on -- the ORDER of the finalists -- and stops when that order stops moving.

Every finalist advances by the same trial block, so a ranking read after a block compares survivors
that had equal search budget. Between two consecutive blocks the ranking is compared two ways, both
of which must hold for the transition to count as stable:

- **pairwise rank agreement**, the share of finalist pairs ordered the same way in both blocks. At
  the default 1.0 this is "identical order"; a looser declaration tolerates churn in the tail while
  still refusing a reshuffle at the top.
- **an unchanged leader**, because a run whose top row keeps swapping has not settled whatever the
  tail does.

Rankings are read from TUNING-split trial values only. Nothing on the final split may enter this
loop -- that is what makes the held-out score a score rather than a stopping criterion.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def ranking_from(objective: Mapping[str, float]) -> tuple[str, ...]:
    """Finalists ordered best-first: higher objective, then name, so ties never flap."""
    return tuple(sorted(objective, key=lambda name: (-objective[name], name)))


def rank_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    """Share of finalist pairs the two rankings order the same way (1.0 = identical order).

    Only names present in BOTH rankings are compared; a name that appears in one alone contributes
    no pair, because there is no ordering of it to agree or disagree about.
    """
    shared = [name for name in left if name in set(right)]
    if len(shared) < 2:
        return 1.0
    right_rank = {name: index for index, name in enumerate(right)}
    left_rank = {name: index for index, name in enumerate(shared)}
    pairs = concordant = 0
    for i, first in enumerate(shared):
        for second in shared[i + 1 :]:
            pairs += 1
            same = (left_rank[first] < left_rank[second]) == (
                right_rank[first] < right_rank[second]
            )
            concordant += int(same)
    return concordant / pairs


@dataclass(frozen=True)
class BlockSnapshot:
    """One trial block: what every finalist had reached, and whether the order held."""

    index: int
    trials_per_finalist: int
    consumed_trials: int
    ranking: tuple[str, ...]
    objective: dict[str, float]
    agreement: float | None
    leader_held: bool | None
    stable: bool
    stable_streak: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "trials_per_finalist": self.trials_per_finalist,
            "consumed_trials": self.consumed_trials,
            "ranking": list(self.ranking),
            "objective": dict(self.objective),
            "agreement": self.agreement,
            "leader_held": self.leader_held,
            "stable": self.stable,
            "stable_streak": self.stable_streak,
        }


def build_snapshot(
    *,
    index: int,
    trials_per_finalist: int,
    consumed_trials: int,
    objective: Mapping[str, float],
    previous: BlockSnapshot | None,
    agreement_floor: float,
) -> BlockSnapshot:
    """Rank this block and score the transition from the one before it.

    The first block has no transition to score, so its agreement and leader check are `None` and
    its streak is zero -- one block can never satisfy a rule about how the order MOVED.
    """
    ranking = ranking_from(objective)
    if previous is None:
        return BlockSnapshot(
            index=index,
            trials_per_finalist=trials_per_finalist,
            consumed_trials=consumed_trials,
            ranking=ranking,
            objective=dict(objective),
            agreement=None,
            leader_held=None,
            stable=False,
            stable_streak=0,
        )
    agreement = rank_agreement(previous.ranking, ranking)
    leader_held = bool(ranking and previous.ranking and ranking[0] == previous.ranking[0])
    stable = agreement >= agreement_floor and leader_held
    return BlockSnapshot(
        index=index,
        trials_per_finalist=trials_per_finalist,
        consumed_trials=consumed_trials,
        ranking=ranking,
        objective=dict(objective),
        agreement=agreement,
        leader_held=leader_held,
        stable=stable,
        stable_streak=previous.stable_streak + 1 if stable else 0,
    )


__all__ = ["BlockSnapshot", "build_snapshot", "rank_agreement", "ranking_from"]
