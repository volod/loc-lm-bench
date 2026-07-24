"""Types and constants of the graph-vector fusion evidence lane.

The lane answers one question the plain `compare-retrieval` table cannot: does adding graph
evidence to the vector lane recover the questions whose answer needs MORE THAN ONE span, and at
what cost to everything else? That needs three things the flat table lacks -- a multi-span metric
(`all_spans_at_k`), uncertainty on a slice of a dozen items, and the item-level paired ledger a
reviewer can actually read.
"""

from typing import TYPE_CHECKING, NamedTuple

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import SourceSpanRecord
from llb.rag.compare import (
    Retriever as Retriever,
)  # the one `.retrieve` seam, re-used not re-declared
from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion_spans import DEFAULT_SPAN_IDENTITY, SPAN_MERGE_MIN_RATIO

if TYPE_CHECKING:  # imported lazily: the floor is opt-in and costs an extra pass per row
    from llb.rag.noise_floor import NoiseFloorReport

# The slice the lane is built to measure; other question types still report as context slices.
FOCUS_SLICE = "multi-hop"
# Overall-recall loss (absolute, at k) still counted as "did not regress" for the verdict.
OVERALL_RECALL_TOLERANCE = 0.0

# Row labels of a graph-weight sweep. One definition, so the sweep that WRITES them and the
# verdict that SELECTS fused candidates from them can never drift apart.
VECTOR_ROW = "vector"
GRAPH_ROW_PREFIX = "graph/"
FUSED_ROW_PREFIX = "fused/"
ROUTED_ROW_PREFIX = "routed/"
# A fused row is identified by BOTH fusion knobs: the graph share and the per-lane candidate
# depth the share is applied over (`/d<depth>`), so a depth sweep and a weight sweep are the same
# table and the verdict ranks across both.
FUSED_ROW_TEMPLATE = FUSED_ROW_PREFIX + "{strategy}@{weight:.2f}/d{depth}"
# A third knob: the span-identity policy the two lanes are fused by. The default policy carries NO
# marker, so an `exact` row keeps the exact label (and therefore the exact comparability) it had
# before the policy existed; only a non-default policy extends the label.
IDENTITY_MARKER = "/i"
# A fourth knob, and a parameter OF the identity policy: the merge threshold a folding policy
# applies. Same rule -- the default value carries no marker, so every row measured before the
# threshold was swept keeps its label.
MERGE_RATIO_MARKER = "/r"


def fused_row_label(
    strategy: str,
    weight: float,
    depth: int,
    span_identity: str = DEFAULT_SPAN_IDENTITY,
    merge_ratio: float = SPAN_MERGE_MIN_RATIO,
) -> str:
    """The one place a fused row label is formatted; `lanes.py` parses exactly this shape back."""
    label = FUSED_ROW_TEMPLATE.format(strategy=strategy, weight=weight, depth=depth)
    if span_identity != DEFAULT_SPAN_IDENTITY:
        label = f"{label}{IDENTITY_MARKER}{span_identity}"
    if merge_ratio != SPAN_MERGE_MIN_RATIO:
        label = f"{label}{MERGE_RATIO_MARKER}{merge_ratio:.2f}"
    return label


def routed_row_label(
    strategy: str,
    weight: float,
    depth: int,
    span_identity: str = DEFAULT_SPAN_IDENTITY,
    merge_ratio: float = SPAN_MERGE_MIN_RATIO,
) -> str:
    """Label a question-type-routed row by its non-zero graph share and fusion knobs."""
    label = fused_row_label(strategy, weight, depth, span_identity, merge_ratio)
    return ROUTED_ROW_PREFIX + label[len(FUSED_ROW_PREFIX) :]


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


class AgreementReport(TypedDict):
    """Cross-lane agreement of one fused row: candidates BOTH lanes returned, per question.

    The number the span-identity policy exists to move, and the reason candidate depth is or is
    not a live knob: under undamped RRF only a candidate both lanes vouch for can be promoted out
    of a deeper pool into the top-k.
    """

    questions: int
    questions_with_shared_candidate: int
    share_of_questions: float
    mean_shared_candidates: float


class RoutingReport(TypedDict):
    """Auditable decision counts for one routed row."""

    graph_questions: int
    vector_questions: int
    sidecar_questions: int
    heuristic_questions: int
    slices: dict[str, dict[str, int]]


class RowReport(TypedDict):
    """One compared retrieval row: overall plus every question-type slice."""

    overall: SliceReport
    slices: dict[str, SliceReport]
    agreement: NotRequired[AgreementReport]
    routing: NotRequired[RoutingReport]


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
    # Measurement floor per swept row, present only when it was asked for
    # (`compare-graph-fusion --noise-floor`). The sweep publishes three-decimal recall@k rows
    # across a dozen weights, so the floor states which of them are separated at all. It is
    # measured over every item AND over the focus slice alone, because the verdict is decided on
    # the focus slice and a floor measured on the whole set does not bound a slice of it.
    noise_floor: NotRequired["NoiseFloorReport"]
    noise_floor_focus: NotRequired["NoiseFloorReport"]  # absent when the focus slice is empty
