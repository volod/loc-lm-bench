"""Types and constants of the graph-vector fusion evidence lane.

The lane answers one question the plain `compare-retrieval` table cannot: does adding graph
evidence to the vector lane recover the questions whose answer needs MORE THAN ONE span, and at
what cost to everything else? That needs three things the flat table lacks -- a multi-span metric
(`all_spans_at_k`), uncertainty on a slice of a dozen items, and the item-level paired ledger a
reviewer can actually read.
"""

from typing import NamedTuple

from typing_extensions import TypedDict

from llb.core.contracts.rag import SourceSpanRecord
from llb.rag.compare import (
    Retriever as Retriever,
)  # the one `.retrieve` seam, re-used not re-declared
from llb.rag.fusion_evidence.stats import Interval, PairedComparison

# The slice the lane is built to measure; other question types still report as context slices.
FOCUS_SLICE = "multi-hop"
# Overall-recall loss (absolute, at k) still counted as "did not regress" for the verdict.
OVERALL_RECALL_TOLERANCE = 0.0

# Row labels of a graph-weight sweep. One definition, so the sweep that WRITES them and the
# verdict that SELECTS fused candidates from them can never drift apart.
VECTOR_ROW = "vector"
GRAPH_ROW_PREFIX = "graph/"
FUSED_ROW_PREFIX = "fused/"
# A fused row is identified by BOTH fusion knobs: the graph share and the per-lane candidate
# depth the share is applied over (`/d<depth>`), so a depth sweep and a weight sweep are the same
# table and the verdict ranks across both.
FUSED_ROW_TEMPLATE = FUSED_ROW_PREFIX + "{strategy}@{weight:.2f}/d{depth}"

METRIC_RECALL = "recall_at_k"
METRIC_ALL_SPANS = "all_spans_at_k"
METRIC_COVERAGE = "span_coverage"
METRIC_MRR = "mrr"
METRICS = (METRIC_RECALL, METRIC_ALL_SPANS, METRIC_COVERAGE, METRIC_MRR)

VERDICT_ADOPT = "adopt"
VERDICT_REJECT = "reject"
# A positive point estimate whose paired interval still includes no difference.
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_NO_EVIDENCE = "no_evidence"


class EvidenceItem(NamedTuple):
    """One scored gold item: its identity, query, labeled spans, and question-type slice."""

    item_id: str
    question: str
    spans: list[SourceSpanRecord]
    question_type: str | None


class SliceReport(TypedDict):
    """One row's metrics over one item slice, with its paired delta against the baseline row."""

    n: int
    metrics: dict[str, Interval]
    paired_vs_baseline: dict[str, PairedComparison]


class RowReport(TypedDict):
    """One compared retrieval row: overall plus every question-type slice."""

    overall: SliceReport
    slices: dict[str, SliceReport]


class ItemOutcome(TypedDict):
    """Item-level paired outcome for the focus slice -- the small-n reviewer view."""

    item_id: str
    question: str
    n_spans: int
    rows: dict[str, dict[str, float]]


class Verdict(TypedDict):
    """Adopt-or-reject decision for the best fused row on the focus slice."""

    focus_slice: str
    focus_n: int
    baseline: str
    best_row: str | None
    decision: str
    reason: str


class FusionEvidenceReport(TypedDict):
    """The full lane artifact: rows, focus-slice item ledger, and the verdict."""

    k: int
    n: int
    baseline: str
    focus_slice: str
    resamples: int
    confidence: float
    seed: int
    rows: dict[str, RowReport]
    focus_items: list[ItemOutcome]
    verdict: Verdict
