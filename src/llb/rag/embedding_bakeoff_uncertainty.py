"""Paired sampling uncertainty for the embedder bake-off: is a candidate's lead an item set?

The bake-off ranks four encoders on ONE gold set, and the gap between the leader and the incumbent
is routinely worth one or two questions. The measurement floor (`llb.rag.noise_floor`) answers only
whether that gap is numeric noise; it cannot answer whether the SAME gap would survive a different
draw of questions. This module supplies the second reading the fusion sweep already has: per
candidate, the per-item metric vector against the incumbent embedder, a paired percentile-bootstrap
delta over SHARED resample index sets, and the win/loss/tie ledger behind it.

Shared index sets (common random numbers) mean every candidate is resampled on the same questions
as the baseline, so the interval is about the DIFFERENCE and not about the two lanes' separate
sampling noise. The statistics themselves are reused wholesale from
`llb.rag.fusion_evidence.stats` -- they take metric vectors, not fusion rows.

Pure and dependency-free: vectors come from the `.retrieve` seam, so the whole lane is unit-tested
with fake stores (no FAISS, no GPU).
"""

from typing_extensions import TypedDict

from llb.core.contracts.rag import RetrievalPair
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PairedComparison,
    bootstrap_index_sets,
    format_interval,
    paired_comparison,
)
from llb.rag.retrieval import recall_at_k, reciprocal_rank

# Metric keys, identical to the `CandidateResult` field names so a row and its interval line up.
METRIC_RECALL = "recall_at_k"
METRIC_MRR = "mrr"
METRICS = (METRIC_RECALL, METRIC_MRR)

# The incumbent the deltas are measured against: the shipped `RunConfig.embedding_model`. A swap
# recommendation is a statement about replacing THIS row, so it is the natural baseline.
DEFAULT_BASELINE_MODEL = "intfloat/multilingual-e5-base"

DECISION_ADOPT = "adopt"
DECISION_RETAIN = "retain"
DECISION_UNDECIDED = "undecided"

# Metric vectors of one candidate: one value per scored item, in item order.
MetricVectors = dict[str, list[float]]


class PairedRow(TypedDict):
    """One candidate's paired delta against the baseline embedder, keyed by metric.

    Same shape as `fusion_evidence.slices.SliceReport["paired_vs_baseline"]` -- a metric-keyed
    mapping rather than fixed fields, so the metric set stays a parameter of the lane.
    """

    baseline: str
    metrics: dict[str, PairedComparison]


class UncertaintySettings(TypedDict):
    """What the intervals were drawn with, so a report is reproducible from its own header."""

    baseline: str | None
    resamples: int
    confidence: float
    seed: int


class BakeoffVerdict(TypedDict):
    """Adopt-or-retain: the one sentence the operator acts on, and why it says that."""

    decision: str  # DECISION_ADOPT | DECISION_RETAIN | DECISION_UNDECIDED
    model: str | None  # the embedder the decision names
    baseline: str | None
    separated: list[str]  # candidates whose paired recall interval clears zero
    reason: str


def item_vectors(pairs: list[RetrievalPair], k: int) -> MetricVectors:
    """Per-item recall@k and reciprocal rank from ONE retrieval pass (means match the row)."""
    return {
        METRIC_RECALL: [recall_at_k(hits, spans, k) for hits, spans in pairs],
        METRIC_MRR: [reciprocal_rank(hits[:k], spans) for hits, spans in pairs],
    }


def paired_rows(
    vectors: dict[str, MetricVectors],
    baseline: str,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, PairedRow]:
    """Paired delta per candidate against `baseline`, over one shared set of resample indexes.

    Returns an empty mapping when the baseline was not scored in this run: there is nothing to be
    paired against, and inventing a different reference row would silently change the question.
    """
    reference = vectors.get(baseline)
    if reference is None:
        return {}
    n = len(reference[METRIC_RECALL])
    index_sets = bootstrap_index_sets(n, resamples, seed)
    return {
        model: {
            "baseline": baseline,
            "metrics": {
                metric: paired_comparison(
                    candidate[metric], reference[metric], index_sets, confidence
                )
                for metric in METRICS
            },
        }
        for model, candidate in vectors.items()
    }


def recall_delta(paired: PairedRow) -> PairedComparison:
    """The recall@k paired delta -- the metric the adopt-or-retain bar is read on."""
    return paired["metrics"][METRIC_RECALL]


def separates_from_baseline(paired: PairedRow) -> bool:
    """True when the paired recall@k interval lies wholly above zero (the adoption bar)."""
    return recall_delta(paired)["delta"]["lo"] > 0.0


def decide_verdict(paired: dict[str, PairedRow], baseline: str | None) -> BakeoffVerdict:
    """Adopt the best separated candidate, else retain the incumbent (never rank on a point gap).

    "Separated" is deliberately the strict reading: the 95% paired interval of the recall@k delta
    excludes zero. A candidate that merely leads on the point estimate is exactly the case this
    lane exists to refuse.
    """
    if baseline is None or not paired:
        return {
            "decision": DECISION_UNDECIDED,
            "model": None,
            "baseline": baseline,
            "separated": [],
            "reason": (
                "the baseline embedder was not scored in this run, so no paired delta is defined"
            ),
        }
    separated = sorted(
        (
            model
            for model, row in paired.items()
            if model != baseline and separates_from_baseline(row)
        ),
        key=lambda model: (-recall_delta(paired[model])["delta"]["mean"], model),
    )
    if not separated:
        return {
            "decision": DECISION_RETAIN,
            "model": baseline,
            "baseline": baseline,
            "separated": [],
            "reason": (
                f"no candidate's paired recall@k interval clears zero against `{baseline}`, "
                "so the ranking is not supported by this item set"
            ),
        }
    winner = separated[0]
    delta = recall_delta(paired[winner])
    return {
        "decision": DECISION_ADOPT,
        "model": winner,
        "baseline": baseline,
        "separated": separated,
        "reason": (
            f"`{winner}` separates from `{baseline}`: paired recall@k delta "
            f"{format_interval(delta['delta'])}, "
            f"{delta['wins']}/{delta['losses']}/{delta['ties']} win/loss/tie, "
            f"sign-test p={delta['sign_test_p']:.3f}"
        ),
    }
