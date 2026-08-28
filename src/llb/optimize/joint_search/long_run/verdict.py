"""Adopt or retain the default model -- the one sentence the confirmation run exists to license.

The decision is on the CALIBRATED paired reading against the declared incumbent, never on the point
gap: a row that merely leads on the held-out mean is exactly the small-sample rank reversal this
lane refuses. It reuses the same separation test, the same minimum-evidence gate, and the same
borderline qualifier every other adopt-or-retain lane in the repo cuts on
(`llb.rag.fusion_evidence`), so a joint-search verdict and an embedder verdict mean the same thing.

Two things ADD to that decision without ever loosening it:

- the public Ukrainian screen: a candidate whose public coverage is missing or partial can still be
  adopted, but the sentence says so, because a default model resting on one private gold set is a
  recommendation with a single point of failure;
- the quality/latency tradeoff: the run's objectives were multi-objective, so a verdict that named
  only the argmax would throw away the half of the answer an operator on a fixed GPU actually acts
  on. The frontier is reported, and a leader that is NOT the fastest row says so explicitly.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llb.optimize.joint_search.long_run.public_tracks import public_note
from llb.optimize.joint_search.long_run.uncertainty import BoardUncertainty
from llb.rag.fusion_evidence.paired import separates
from llb.rag.fusion_evidence.stability import borderline_note

DECISION_ADOPT = "adopt"
DECISION_RETAIN = "retain"
DECISION_UNDECIDED = "undecided"


@dataclass(frozen=True)
class AdoptionVerdict:
    """The decision, the row it names, and everything that qualifies it."""

    decision: str
    model: str | None
    row: str | None
    incumbent: str | None
    reason: str
    separated: list[str]
    borderline: list[str]
    quality_leader: str | None
    latency_leader: str | None
    frontier: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "model": self.model,
            "row": self.row,
            "incumbent": self.incumbent,
            "reason": self.reason,
            "separated": list(self.separated),
            "borderline": list(self.borderline),
            "quality_leader": self.quality_leader,
            "latency_leader": self.latency_leader,
            "pareto_frontier": list(self.frontier),
            "tradeoff": self.tradeoff,
        }

    @property
    def tradeoff(self) -> str:
        """The quality-versus-latency sentence, stated whether or not the two agree."""
        if self.quality_leader is None:
            return "no scored row, so there is no quality/latency tradeoff to state"
        if self.latency_leader is None or self.quality_leader == self.latency_leader:
            return (
                f"`{self.quality_leader}` leads on quality AND is the fastest row on the frontier, "
                "so quality and latency select the same configuration"
            )
        return (
            f"`{self.quality_leader}` leads on quality while `{self.latency_leader}` is the "
            "fastest row on the frontier: an operator trading quality for latency should take the "
            "second, and the recommendation above is the quality side of that frontier"
        )


def _row_model(row: str) -> str:
    return row.split("::", 1)[0]


def _leaders(uncertainty: BoardUncertainty) -> tuple[str | None, str | None]:
    """The frontier's best-quality row and its lowest-latency row (both point estimates)."""
    frontier = [row for row in uncertainty.frontier if row in uncertainty.quality]
    if not frontier:
        return None, None
    best_quality = max(frontier, key=lambda row: (uncertainty.quality[row]["mean"], row))
    fastest = min(frontier, key=lambda row: (uncertainty.latency[row]["mean"], row))
    return best_quality, fastest


def decide(
    uncertainty: BoardUncertainty,
    *,
    incumbent: str | None,
    public: Mapping[str, Any],
) -> AdoptionVerdict:
    """Adopt the best SEPARATED candidate, else retain the incumbent; qualify either way."""
    quality_leader, latency_leader = _leaders(uncertainty)
    reading = _read_board(uncertainty, incumbent=incumbent, public=public)
    return AdoptionVerdict(
        decision=reading.decision,
        model=reading.model,
        row=reading.row,
        incumbent=incumbent,
        reason=reading.reason,
        separated=reading.separated,
        borderline=reading.borderline,
        quality_leader=quality_leader,
        latency_leader=latency_leader,
        frontier=list(uncertainty.frontier),
    )


@dataclass(frozen=True)
class _Reading:
    """The decision half of a verdict, before the frontier context is attached to it."""

    decision: str
    model: str | None
    row: str | None
    reason: str
    separated: list[str]
    borderline: list[str]


def _read_board(
    uncertainty: BoardUncertainty, *, incumbent: str | None, public: Mapping[str, Any]
) -> _Reading:
    """Which row the calibrated paired readings license, and why."""
    if incumbent is None or uncertainty.baseline is None:
        return _Reading(
            decision=DECISION_UNDECIDED,
            model=None,
            row=None,
            reason=(
                "no incumbent was declared for this board"
                if incumbent is None
                else f"the declared incumbent `{incumbent}` was not scored on the held-out split, "
                "so no paired delta against it is defined"
            ),
            separated=[],
            borderline=[],
        )
    confidence = uncertainty.confidence
    separated = sorted(
        row for row, comparison in uncertainty.paired.items() if separates(comparison, confidence)
    )
    borderline = sorted(
        row
        for row, comparison in uncertainty.paired.items()
        if (stability := comparison.get("stability")) and stability["borderline"]
    )
    if not separated:
        return _Reading(
            decision=DECISION_RETAIN,
            model=_row_model(uncertainty.baseline),
            row=uncertainty.baseline,
            reason=(
                f"no candidate separates from `{incumbent}` on the {uncertainty.n_items}-item "
                f"held-out split at {confidence:.0%}, so the point-estimate ranking is not "
                "supported by this item set"
                + _borderline_clause(uncertainty, borderline)
                + public_note(public, incumbent)
            ),
            separated=[],
            borderline=borderline,
        )
    winner = max(separated, key=lambda row: (uncertainty.paired[row]["delta"]["mean"], row))
    delta = uncertainty.paired[winner]["delta"]
    return _Reading(
        decision=DECISION_ADOPT,
        model=_row_model(winner),
        row=winner,
        reason=(
            f"`{winner}` separates from `{incumbent}` by {delta['mean']:+.3f} "
            f"[{delta['lo']:+.3f}, {delta['hi']:+.3f}] on the {uncertainty.n_items}-item held-out "
            f"split at {confidence:.0%}"
            + _borderline_clause(uncertainty, [row for row in borderline if row == winner])
            + public_note(public, _row_model(winner))
        ),
        separated=separated,
        borderline=borderline,
    )


def _borderline_clause(uncertainty: BoardUncertainty, rows: Sequence[str]) -> str:
    """Say which named rows would read differently at a neighbouring conventional level."""
    return borderline_note([(row, uncertainty.paired[row].get("stability")) for row in rows])


__all__ = [
    "DECISION_ADOPT",
    "DECISION_RETAIN",
    "DECISION_UNDECIDED",
    "AdoptionVerdict",
    "decide",
]
