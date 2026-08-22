"""Where the model and the shipped constant agree, and the cut that would reproduce today.

The constant's decision is per ITEM -- this drafted question is a repeat of something already
seen -- so the model is read the same way: a candidate's score is its best pair against a record
that precedes it. Two numbers then say everything about how the two policies relate: the lowest
score among the items the constant dropped, and the highest score among the items it kept. While
the first is above the second a cut exists that reproduces today's decisions exactly, and the
widest-margin one is the provisional operating point this lane publishes.

Those comparisons are made on the MATCH WEIGHT, not the probability. A well-separated fit pushes
both a duplicate and a merely-similar item to a probability that rounds to 1.0, so the margin
between them survives only in the log-odds the probability was computed from. The cut is published
in both forms, with a flag saying whether the probability form still reproduces the decisions --
the seam's own threshold is a probability, so a cut that saturates is a fact an operator needs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.goldset.schema import GoldItem
from llb.linkage.constants import DEFAULT_MATCH_THRESHOLD
from llb.linkage.model import LinkagePair
from llb.prep.ontology.linkage.agreements import best_parent, other_id, pair_payload
from llb.prep.ontology.linkage.constants import ROLE_CANDIDATE

_ROUND = 6
# A well-separated fit puts every near-certain pair within a rounding error of 1.0, so a
# probability is only readable beside its weight when it keeps more decimals than a weight needs.
_PROBABILITY_ROUND = 12
_NO_DROPS = "no drop to price: the seam's default cut stands"
_DEFAULT_HOLDS = (
    "the seam's default cut, which already drops exactly what the shipped constant drops"
)
_TIGHTEST = (
    "the lowest-scoring dropped item: the tightest cut that preserves every shipped drop, because "
    "the seam's default cut does not"
)
_OVERLAPPING = (
    "the lowest-scoring dropped item: the model does not separate today's decisions, so no cut "
    "reproduces them exactly"
)


def weight_of(probability: float) -> float:
    """A match probability as its match weight (log2 odds) -- the seam's own relation."""
    if probability >= 1.0:
        return float("inf")
    if probability <= 0.0:
        return float("-inf")
    from math import log2

    return log2(probability / (1.0 - probability))


def probability_of(weight: float) -> float:
    """A match weight back as a match probability."""
    if weight == float("inf"):
        return 1.0
    return float(1.0 / (1.0 + 2.0**-weight))


@dataclass(frozen=True)
class ShadowDecisions:
    """Both policies' verdict on every candidate, plus the pair each verdict rests on."""

    constant_dropped: set[str]
    candidate_score: dict[str, LinkagePair | None]
    scored_drops: dict[str, LinkagePair]
    unscored_drops: tuple[str, ...]
    questions: dict[str, str]
    partners: dict[str, JsonObject]
    labels: dict[str, dict[int, str]]

    @classmethod
    def build(
        cls,
        *,
        shipped: dict[str, tuple[str, str] | None],
        candidates: Sequence[GoldItem],
        by_item: dict[tuple[str, str], str],
        positions: dict[str, int],
        pairs: dict[tuple[str, str], LinkagePair],
        neighbours: dict[str, list[LinkagePair]],
        item_of: dict[str, GoldItem],
        labels: dict[str, dict[int, str]],
    ) -> "ShadowDecisions":
        score: dict[str, LinkagePair | None] = {}
        partners: dict[str, JsonObject] = {}
        for item in candidates:
            record = by_item.get((ROLE_CANDIDATE, item.id))
            pair = best_parent(record, neighbours, positions) if record is not None else None
            score[item.id] = pair
            if pair is not None and record is not None:
                partners[item.id] = _partner(item_of[other_id(pair, record)])
        scored = {
            item_id: pairs[key]
            for item_id, key in shipped.items()
            if key is not None and key in pairs
        }
        return cls(
            constant_dropped=set(shipped),
            candidate_score=score,
            scored_drops=scored,
            unscored_drops=tuple(sorted(set(shipped) - set(scored))),
            questions={item.id: item.question for item in candidates},
            partners=partners,
            labels=labels,
        )

    @property
    def n_shipped(self) -> int:
        return len(self.constant_dropped)

    @property
    def n_scored(self) -> int:
        return len(self.scored_drops)

    @property
    def drop_weights(self) -> list[float]:
        return [
            pair.match_weight
            for item_id in self.constant_dropped
            if (pair := self.candidate_score.get(item_id)) is not None
        ]

    @property
    def kept_weights(self) -> list[float]:
        return [
            pair.match_weight
            for item_id, pair in self.candidate_score.items()
            if pair is not None and item_id not in self.constant_dropped
        ]

    @property
    def unreachable_drops(self) -> tuple[str, ...]:
        """Items the constant dropped that no scored pair can reproduce at ANY cut."""
        return tuple(
            sorted(
                item_id
                for item_id in self.constant_dropped
                if self.candidate_score.get(item_id) is None
            )
        )

    @property
    def separates(self) -> bool:
        """True when a cut exists that drops exactly what the constant dropped."""
        if not self.drop_weights or self.unreachable_drops:
            return False
        return min(self.drop_weights) > max(self.kept_weights, default=float("-inf"))

    @property
    def margin(self) -> float | None:
        """How much match weight separates the lowest drop from the highest keep.

        None when either side is empty: a margin against nothing is not a number, and writing one
        into an artifact (as an infinity JSON cannot even carry) would read as a measurement.
        """
        if not self.drop_weights or not self.kept_weights:
            return None
        return min(self.drop_weights) - max(self.kept_weights)

    @property
    def provisional_weight(self) -> float:
        """The cut this lane publishes, never above the lowest-scoring drop.

        The seam's default cut stands whenever it already decides what the constant decided --
        that is the whole point of a shadow lane, and moving a default off an unsupervised fit is
        what the reviewer-labelled set is for. Otherwise the cut is the tightest one that keeps
        every shipped drop, which is stable in a way the midpoint of two weights is not: non-match
        weights run to tens of negative bits, so their arithmetic midpoint lands far below any
        value an operator would adopt and moves with every unrelated pair added to the table.
        """
        if not self.drop_weights:
            return weight_of(DEFAULT_MATCH_THRESHOLD)
        lowest_drop = min(self.drop_weights)
        default = weight_of(DEFAULT_MATCH_THRESHOLD)
        if (
            self.separates
            and max(self.kept_weights, default=float("-inf")) < default <= lowest_drop
        ):
            return default
        return lowest_drop

    @property
    def probability_cut_reproduces(self) -> bool:
        """Whether the same cut, expressed as a PROBABILITY, still decides what the weight does."""
        cut = probability_of(self.provisional_weight)
        drops = [probability_of(weight) for weight in self.drop_weights]
        kept = [probability_of(weight) for weight in self.kept_weights]
        if not drops:
            return True
        return min(drops) >= cut and max(kept, default=0.0) < cut

    def thresholds_payload(self) -> JsonObject:
        basis = self._basis()
        return {
            "provisional_match_weight": _rounded(self.provisional_weight),
            "provisional_match_probability": _rounded(
                probability_of(self.provisional_weight), _PROBABILITY_ROUND
            ),
            "provisional_threshold_basis": basis,
            "separates_shipped_decisions": self.separates,
            "probability_cut_reproduces_shipped_drops": self.probability_cut_reproduces,
            "shipped_drop_weight_min": _rounded(min(self.drop_weights, default=None)),
            "kept_weight_max": _rounded(max(self.kept_weights, default=None)),
            "separation_margin_weight": _rounded(self.margin),
            "linkage_default_threshold": DEFAULT_MATCH_THRESHOLD,
            "unreproducible_drop_ids": list(self.unreachable_drops),
            "unscored_drop_ids": list(self.unscored_drops),
        }

    def _basis(self) -> str:
        if not self.drop_weights:
            return _NO_DROPS
        if not self.separates:
            return _OVERLAPPING
        return (
            _DEFAULT_HOLDS
            if self.provisional_weight == weight_of(DEFAULT_MATCH_THRESHOLD)
            else _TIGHTEST
        )

    def model_drops(self, weight: float) -> set[str]:
        return {
            item_id
            for item_id, pair in self.candidate_score.items()
            if pair is not None and pair.match_weight >= weight
        }

    def point(self, name: str, weight: float) -> JsonObject:
        """What one candidate cut would decide, and every item the two policies disagree on."""
        dropped = self.model_drops(weight)
        disagreeing = sorted(dropped ^ self.constant_dropped)
        return {
            "name": name,
            "match_weight": _rounded(weight),
            "match_probability": _rounded(probability_of(weight), _PROBABILITY_ROUND),
            "n_model_drops": len(dropped),
            "n_constant_drops": len(self.constant_dropped),
            "n_agree": len(self.candidate_score) - len(disagreeing),
            "n_disagree": len(disagreeing),
            "disagreements": [self._row(item_id, item_id in dropped) for item_id in disagreeing],
        }

    def _row(self, item_id: str, model_dropped: bool) -> JsonObject:
        pair = self.candidate_score.get(item_id)
        row: JsonObject = {
            "id": item_id,
            "question": self.questions.get(item_id, ""),
            "model": "drop" if model_dropped else "keep",
            "constant": "drop" if item_id in self.constant_dropped else "keep",
            "nearest": self.partners.get(item_id),
        }
        if pair is not None:
            row.update(pair_payload(pair, self.labels))
        return row


def operating_points(decisions: ShadowDecisions) -> list[JsonObject]:
    """The two cuts a reviewer compares: the provisional one, and the seam's default."""
    default = weight_of(DEFAULT_MATCH_THRESHOLD)
    points = [decisions.point("provisional", decisions.provisional_weight)]
    if decisions.provisional_weight != default:
        points.append(decisions.point("linkage-default", default))
    return points


def _partner(item: GoldItem) -> JsonObject:
    return {"id": item.id, "question": item.question, "source_doc_id": item.source_doc_id}


def _rounded(value: Any, digits: int = _ROUND) -> float | None:
    return None if value is None else round(float(value), digits)
