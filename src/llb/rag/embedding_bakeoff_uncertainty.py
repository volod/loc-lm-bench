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
from llb.rag.fusion_evidence.stability import (
    ReadingStability,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PairedComparison,
    bootstrap_index_sets,
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

# Adoption bars: the paired metric interval(s) a candidate must clear to be adopted.
#
# `recall_at_k` is the DEFAULT and the only unconditional bar -- an encoder that finds evidence the
# incumbent misses helps every retrieval configuration. `mrr` is the second, explicitly-SCOPED bar:
# a first-hit-rank gain is worth adopting only where rank is binding (a small `top_k`, or a
# cross-encoder reranker that can only re-sort what it is handed). At a generous `top_k` with room
# in the context budget, ranking the same evidence earlier changes nothing the answer can see, so
# the bar stays OFF unless the operator's configuration puts it in scope -- and the end-to-end
# evidence for that scope lives in the embedder bake-off section of `docs/impl/current/rag-core.md`.
BAR_RECALL = METRIC_RECALL
BAR_FIRST_HIT = METRIC_MRR
BARS = (BAR_RECALL, BAR_FIRST_HIT)
DEFAULT_BARS = (BAR_RECALL,)

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
    bars: list[str]
    resamples: int
    confidence: float
    seed: int


class BakeoffVerdict(TypedDict):
    """Adopt-or-retain: the one sentence the operator acts on, and why it says that."""

    decision: str  # DECISION_ADOPT | DECISION_RETAIN | DECISION_UNDECIDED
    model: str | None  # the embedder the decision names
    baseline: str | None
    separated: list[str]  # candidates clearing at least one enabled bar
    bars: list[str]  # the adoption bars this verdict was decided on
    cleared: dict[str, list[str]]  # per separated candidate, which bars it cleared
    # Per candidate, the enabled bars whose separation reading a neighbouring conventional
    # confidence level would change. A qualifier on the decision, never a third outcome: a
    # borderline bar still counts exactly where it counted.
    borderline: dict[str, list[str]]
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


def bar_stability(paired: PairedRow, bar: str) -> ReadingStability | None:
    """How settled one bar's separation reading is, or None on an artifact carrying no draw."""
    return paired["metrics"][bar].get("stability")


def recall_delta(paired: PairedRow) -> PairedComparison:
    """The recall@k paired delta -- the metric the unconditional adoption bar is read on."""
    return paired["metrics"][METRIC_RECALL]
