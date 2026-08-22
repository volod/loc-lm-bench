"""What a linkage run produces: counted blocks, fitted parameters, scored pairs, clusters."""

from collections.abc import Container, Sequence
from dataclasses import dataclass, field
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.linkage.spec import LinkageSpec

# Splink emits a "<column> is NULL" level for every comparison; it has no m/u by construction.
_NULL_LEVEL_SUFFIX = "is NULL"

# (model said match, reviewer said match) -> confusion-matrix cell.
_OUTCOME = {
    (True, True): "tp",
    (True, False): "fp",
    (False, False): "tn",
    (False, True): "fn",
}


@dataclass(frozen=True)
class BlockingCount:
    """How many comparisons one blocking rule generates -- recorded BEFORE the fit.

    A rule that generates more pairs than the host can score is a run that dies halfway through,
    so the count is an artifact in its own right rather than a log line.
    """

    rule: str
    pre_filter: int
    post_filter: int

    def payload(self) -> JsonObject:
        return {
            "rule": self.rule,
            "comparisons_pre_filter": self.pre_filter,
            "comparisons_post_filter": self.post_filter,
        }


@dataclass(frozen=True)
class MatchParameter:
    """The fitted m and u of one comparison level: P(agree | match) and P(agree | non-match)."""

    comparison: str
    level: str
    m: float | None
    u: float | None

    def payload(self) -> JsonObject:
        return {"comparison": self.comparison, "level": self.level, "m": self.m, "u": self.u}


@dataclass(frozen=True)
class LinkagePair:
    """One scored candidate pair: the probability plus the per-level agreement behind it."""

    left_id: str
    right_id: str
    match_probability: float
    match_weight: float
    agreement: dict[str, int] = field(default_factory=dict)

    def payload(self) -> JsonObject:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "match_probability": self.match_probability,
            "match_weight": self.match_weight,
            "agreement": dict(self.agreement),
        }


@dataclass(frozen=True)
class LinkageCluster:
    """A proposed identity: the records connected at the run's threshold.

    Proposed, never applied -- adopting a merge is a separate, evidence-gated decision.
    """

    cluster_id: str
    record_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.record_ids)

    def payload(self) -> JsonObject:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "record_ids": list(self.record_ids),
        }


@dataclass(frozen=True)
class AccuracyPoint:
    """One row of the labelled accuracy curve: what a candidate threshold would decide.

    This is the artifact an operating threshold is READ OFF, rather than picked -- and it exists
    only where a reviewer-labelled set does. Without labels the seam publishes a ranked candidate
    list and no operating point.
    """

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def payload(self) -> JsonObject:
        return {
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class LinkageResult:
    """One whole identity decision, ready to be written into a run bundle."""

    spec: LinkageSpec
    blocking_counts: tuple[BlockingCount, ...]
    match_parameters: tuple[MatchParameter, ...]
    pairs: tuple[LinkagePair, ...]
    clusters: tuple[LinkageCluster, ...]
    trained_model: JsonObject
    n_records: int
    trained_from_labels: bool = False
    accuracy: tuple[AccuracyPoint, ...] = ()
    # What the run's OWN cut scored against the labels, computed from the decisions rather than
    # read off the curve -- curve rows sit at rounded match weights, and the number an operator
    # acts on should not carry that rounding. `pair_operating_point` scores the pairwise cut;
    # `cluster_operating_point` scores what the clustering actually merged, which is wider.
    pair_operating_point: AccuracyPoint | None = None
    cluster_operating_point: AccuracyPoint | None = None

    @property
    def matched_pairs(self) -> tuple[LinkagePair, ...]:
        """Pairs at or above the spec's threshold -- the ones the clustering acted on."""
        return tuple(p for p in self.pairs if p.match_probability >= self.spec.match_threshold)

    @property
    def untrained_levels(self) -> tuple[MatchParameter, ...]:
        """Comparison levels the fit could not estimate an m or u for.

        Not an error: a reviewer label set covers only the levels its matches exhibit, so a
        label-fitted model legitimately leaves the rest at Splink's defaults. It is reported
        rather than hidden, because a level with no estimate is a probability nobody measured.
        """
        return tuple(
            p
            for p in self.match_parameters
            if not p.level.endswith(_NULL_LEVEL_SUFFIX) and (p.m is None or p.u is None)
        )

    @property
    def multi_record_clusters(self) -> tuple[LinkageCluster, ...]:
        return tuple(c for c in self.clusters if c.size > 1)

    def cluster_of(self, record_id: str) -> str | None:
        for cluster in self.clusters:
            if record_id in cluster.record_ids:
                return cluster.cluster_id
        return None

    def summary(self) -> JsonObject:
        """The numbers an operator reads first; also the run bundle's metadata block."""
        return {
            "n_records": self.n_records,
            "n_scored_pairs": len(self.pairs),
            "n_matched_pairs": len(self.matched_pairs),
            "n_clusters": len(self.clusters),
            "n_multi_record_clusters": len(self.multi_record_clusters),
            "largest_cluster": max((c.size for c in self.clusters), default=0),
            "match_threshold": self.spec.match_threshold,
            "seed": self.spec.seed,
            "trained_from_labels": self.trained_from_labels,
            "n_accuracy_points": len(self.accuracy),
            "n_untrained_levels": len(self.untrained_levels),
        }

    def best_accuracy(self) -> AccuracyPoint | None:
        """The curve's highest-F1 row -- the operating point a labelled set actually supports.

        A recommendation, not an adoption: the threshold a run USES is the spec's, and moving it
        is a decision taken on retrieval or review evidence with this row as the input.
        """
        if not self.accuracy:
            return None
        return max(self.accuracy, key=lambda point: (point.f1, point.precision, -point.threshold))


def score_labels(
    merged: Container[tuple[str, str]], labels: Sequence[Any], threshold: float
) -> AccuracyPoint:
    """Score a set of merge decisions against reviewer labels: exact counts, no rounding.

    A labelled pair no blocking rule generated is scored as a non-match, which is what the run
    actually decided about it -- a pair never compared is a merge never proposed.
    """
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for label in labels:
        predicted = _key(label.left_id, label.right_id) in merged
        counts[_OUTCOME[(predicted, bool(label.match))]] += 1
    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    return AccuracyPoint(
        threshold=threshold,
        true_positives=counts["tp"],
        false_positives=counts["fp"],
        true_negatives=counts["tn"],
        false_negatives=counts["fn"],
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
    )


def pairs_above(pairs: Sequence[LinkagePair], threshold: float) -> set[tuple[str, str]]:
    """The pairs the model scored at or above a cut -- the PAIRWISE merge decision."""
    return {
        _key(pair.left_id, pair.right_id) for pair in pairs if pair.match_probability >= threshold
    }


def pairs_co_clustered(clusters: Sequence[LinkageCluster]) -> set[tuple[str, str]]:
    """Every pair the clustering put in one identity -- the run's ACTUAL merge decision.

    Wider than `pairs_above`: connected components merge a pair scoring below the cut when a third
    record links them, which is exactly what a single pairwise threshold cannot do. Reporting both
    is what keeps the difference visible rather than folded into one number.
    """
    return {
        _key(left, right)
        for cluster in clusters
        for index, left in enumerate(cluster.record_ids)
        for right in cluster.record_ids[index + 1 :]
    }


def _key(left: str, right: str) -> tuple[str, str]:
    """Pair identity is unordered: a reviewer's (a, b) is the model's (b, a)."""
    return (left, right) if left <= right else (right, left)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def sorted_pairs(pairs: Any) -> tuple[LinkagePair, ...]:
    """Stable pair order: highest probability first, ties broken by record id."""
    return tuple(sorted(pairs, key=lambda p: (-p.match_probability, p.left_id, p.right_id)))


def sorted_clusters(clusters: Any) -> tuple[LinkageCluster, ...]:
    """Stable cluster order: largest first, ties broken by cluster id."""
    return tuple(sorted(clusters, key=lambda c: (-c.size, c.cluster_id)))
