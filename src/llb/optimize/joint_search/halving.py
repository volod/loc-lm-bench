"""Successive-halving schedule helpers (pure; no I/O)."""

from dataclasses import asdict, dataclass
from typing import Sequence

from llb.optimize.joint_search.constants import DEFAULT_ETA, DEFAULT_MIN_FINALISTS
from llb.optimize.tuning_space import TUNING_SPLIT


@dataclass(frozen=True)
class ScreenScore:
    """One candidate's cheap screen measurement on the tuning split."""

    name: str
    quality: float
    latency_s: float = 0.0
    backend: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HalvingRound:
    """One successive-halving round: scores, survivors, and eliminations."""

    round_index: int
    case_limit: int
    split: str
    scores: tuple[ScreenScore, ...]
    kept: tuple[str, ...]
    eliminated: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "case_limit": self.case_limit,
            "split": self.split,
            "scores": [s.to_dict() for s in self.scores],
            "kept": list(self.kept),
            "eliminated": list(self.eliminated),
        }


@dataclass(frozen=True)
class HalvingLedger:
    """Full successive-halving trail; screen/elim always use the tuning split."""

    eta: int
    min_finalists: int
    split: str
    rounds: tuple[HalvingRound, ...]
    finalists: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "eta": self.eta,
            "min_finalists": self.min_finalists,
            "split": self.split,
            "rounds": [r.to_dict() for r in self.rounds],
            "finalists": list(self.finalists),
        }


def rank_scores(scores: Sequence[ScreenScore]) -> list[ScreenScore]:
    """Deterministic ranking: higher quality first, then name ascending."""
    return sorted(scores, key=lambda s: (-s.quality, s.name))


def keep_count(n_candidates: int, *, eta: int, min_keep: int) -> int:
    """How many survivors advance after one halving step."""
    if n_candidates <= 0:
        return 0
    if eta < 2:
        raise ValueError(f"eta must be >= 2, got {eta}")
    return max(min_keep, n_candidates // eta)


def partition_survivors(
    scores: Sequence[ScreenScore],
    *,
    eta: int = DEFAULT_ETA,
    min_keep: int = DEFAULT_MIN_FINALISTS,
) -> tuple[list[str], list[str]]:
    """Split ranked scores into kept / eliminated name lists for one round."""
    ranked = rank_scores(scores)
    if not ranked:
        return [], []
    n_keep = keep_count(len(ranked), eta=eta, min_keep=min_keep)
    if n_keep >= len(ranked):
        return [s.name for s in ranked], []
    kept = [s.name for s in ranked[:n_keep]]
    eliminated = [s.name for s in ranked[n_keep:]]
    return kept, eliminated


def screen_limit_for_round(base_limit: int, round_index: int, *, eta: int = DEFAULT_ETA) -> int:
    """Increasing case budget: ``base_limit * eta**round`` (classic SHA)."""
    if base_limit < 1:
        raise ValueError(f"base_limit must be >= 1, got {base_limit}")
    if round_index < 0:
        raise ValueError(f"round_index must be >= 0, got {round_index}")
    return int(base_limit * (eta**round_index))


def build_halving_round(
    scores: Sequence[ScreenScore],
    *,
    round_index: int,
    case_limit: int,
    eta: int = DEFAULT_ETA,
    min_keep: int = DEFAULT_MIN_FINALISTS,
    split: str = TUNING_SPLIT,
) -> HalvingRound:
    """Record one round; ``split`` must stay on the tuning side of the leak fence."""
    if split != TUNING_SPLIT:
        raise ValueError(
            f"halving rounds must use split={TUNING_SPLIT!r}; got {split!r} "
            "(final-split scores belong only on the scoreboard)"
        )
    kept, eliminated = partition_survivors(scores, eta=eta, min_keep=min_keep)
    return HalvingRound(
        round_index=round_index,
        case_limit=case_limit,
        split=split,
        scores=tuple(rank_scores(scores)),
        kept=tuple(kept),
        eliminated=tuple(eliminated),
    )


def finalize_ledger(
    rounds: Sequence[HalvingRound],
    *,
    eta: int = DEFAULT_ETA,
    min_finalists: int = DEFAULT_MIN_FINALISTS,
) -> HalvingLedger:
    """Seal the ledger; finalists are the last round's kept names (or empty)."""
    if not rounds:
        return HalvingLedger(
            eta=eta,
            min_finalists=min_finalists,
            split=TUNING_SPLIT,
            rounds=(),
            finalists=(),
        )
    last = rounds[-1]
    return HalvingLedger(
        eta=eta,
        min_finalists=min_finalists,
        split=TUNING_SPLIT,
        rounds=tuple(rounds),
        finalists=last.kept,
    )
