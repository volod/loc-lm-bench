"""One ranking contract for every machine and rendered view of a decision group."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionStake:
    """The policy-free inputs that order an audit's decision groups."""

    group_index: int
    decide_rows: int
    rows: int
    top_score: float


def stake_key(stake: DecisionStake) -> tuple[int, int, float, int]:
    """Rank work first, then size, score, and the stable file-order id."""
    return (-stake.decide_rows, -stake.rows, -stake.top_score, stake.group_index)


def stake_ranks(stakes: list[DecisionStake]) -> dict[int, int]:
    """Map each file-order group index to its one-based rendered rank."""
    return {
        stake.group_index: rank for rank, stake in enumerate(sorted(stakes, key=stake_key), start=1)
    }
