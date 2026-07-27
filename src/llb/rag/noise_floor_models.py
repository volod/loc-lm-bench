"""Contracts for retrieval measurement-floor reports."""

from typing_extensions import NotRequired, TypedDict


class MetricSpread(TypedDict):
    """Spread of one metric across the jitter replicates, plus its unjittered value."""

    base: float
    min: float
    max: float
    mean: float
    std: float
    half_width: float  # (max - min) / 2 -- the "+/-" to read beside the metric


class LaneFloor(TypedDict):
    """One lane's metric bands plus the fragility that explains how wide they are."""

    recall_at_k: MetricSpread
    mrr: MetricSpread
    n: int
    fragile_items: int  # items whose rank-k and rank-(k+1) scores sit within `jitter`


class FloorMargin(TypedDict):
    """The reading a recommendation rests on: the top two lanes and their gap versus the floor.

    Only the two best lanes matter, because a recommendation names ONE lane: if the leader's gap
    over the runner-up is inside the floor, the report has not distinguished them, whatever the
    third decimal says.
    """

    leader: str
    runner_up: str | None
    delta: float  # leader recall@k - runner-up recall@k (0.0 when there is no runner-up)
    floor: float  # the floor the delta is read against (`floor_recall_at_k`)
    clears_floor: bool
    clearance: NotRequired[float]  # signed distance from the cut: `delta - floor`
    floor_multiple: NotRequired[float | None]  # `delta / floor`; null at a zero floor


class NoiseFloorReport(TypedDict):
    """Per-lane metric spread under score noise, plus the worst-lane floor per metric."""

    replicates: int
    jitter: float
    candidates: int
    seed: int
    lanes: dict[str, LaneFloor]
    unscored: list[str]  # lanes whose candidates expose no score, so nothing can be perturbed
    floor_recall_at_k: float
    floor_mrr: float
    margin: NotRequired[FloorMargin]  # absent when no lane was measured
