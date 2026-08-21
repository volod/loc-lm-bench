"""Types, constants, and the grid parser of the per-hop multi-hop retrievability probe.

The lane answers one question `compare-graph-fusion` cannot: when a two-hop item fails
`all-spans@k`, WHY does it fail? Every ranking knob measured so far leaves the multi-hop
`all-spans@10` ceiling where it was, so the remaining explanations are not about ranking:
either the missing hop is retrievable by the question and simply sits below the cut (a BUDGET
problem, fixed by a larger k or by a second retrieval pass), or the question's own wording never
reaches it at any depth (a QUERY problem, fixed by decomposition), or nothing reaches it at all.
Those lead to opposite fixes, so the probe names which one the corpus supports.
"""

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import SourceSpanRecord as SourceSpanRecord
from llb.rag.comparison.models import Retriever as Retriever  # the one `.retrieve` seam, re-used
from llb.rag.fusion_evidence.models import EvidenceItem as EvidenceItem
from llb.rag.fusion_evidence.stats import Interval

# Budgets the `all-spans@k` curve is read at. 10 is the shipped `top_k` every recorded fusion row
# was measured at; 25 and 50 are the two "would a bigger pool carry both hops" points.
DEFAULT_BUDGETS = (10, 25, 50)
# How deep a hop is searched for before it counts as unreachable BY THE QUESTION. Far past any
# budget an operator would serve, because the point is to separate "below the cut" from "absent".
DEFAULT_PROBE_DEPTH = 200

# Per-item diagnoses, in precedence order (the worst hop decides the item).
DIAGNOSIS_COVERED = "covered"
DIAGNOSIS_BUDGET = "budget"
DIAGNOSIS_QUERY = "query"
DIAGNOSIS_UNREACHABLE = "unreachable"
DIAGNOSES = (DIAGNOSIS_COVERED, DIAGNOSIS_BUDGET, DIAGNOSIS_QUERY, DIAGNOSIS_UNREACHABLE)

# What the counted diagnoses support, as the report states it in one line.
EXPLANATION_BUDGET = "budget"
EXPLANATION_QUERY = "query"
EXPLANATION_UNREACHABLE = "unreachable"
EXPLANATION_MIXED = "mixed"
EXPLANATION_NONE = "no_failing_item"

# Ledger bucket for an item no compared depth carries both hops for.
BUDGET_BUCKET_BEYOND = "beyond"


class HopOutcome(TypedDict):
    """One labeled span of one item, ranked twice: by the item's question and by its own text.

    `span_query_rank` is the retrievability CONTROL. The span text is a verbatim slice of the
    chunk that must be retrieved, so it is the most favorable query that hop can ever be given:
    when the question cannot reach a hop the span text reaches at rank 1, the gap is the query,
    not the index.
    """

    span_index: int
    doc_id: str
    char_start: int
    char_end: int
    n_chars: int
    question_rank: int | None
    span_query_rank: int | None


class ItemBudgetOutcome(TypedDict):
    """One item's multi-span coverage at one retrieval budget, retrieved AT that budget."""

    k: int
    covered_spans: int
    span_coverage: float
    all_spans_at_k: float
    recall_at_k: float


class ItemProbe(TypedDict):
    """One gold item's per-hop ranks, its coverage curve, and the diagnosis they imply."""

    item_id: str
    question: str
    question_type: str | None
    n_spans: int
    hops: list[HopOutcome]
    budgets: list[ItemBudgetOutcome]
    limiting_rank: int | None
    min_budget: int | str
    diagnosis: str
    query_prep: NotRequired[dict[str, object]]


class BudgetReport(TypedDict):
    """One budget's aggregate over one item slice: the curve the diagnosis is read against."""

    k: int
    n: int
    all_spans_at_k: Interval
    span_coverage: float
    recall_at_k: float
    hop_hit_rate: float
    span_query_hop_hit_rate: float


class DiagnosisReport(TypedDict):
    """Counted per-item diagnoses over one slice, plus the explanation they support."""

    n: int
    counts: dict[str, int]
    failing_items: int
    budget_histogram: dict[str, int]
    explanation: str
    reason: str


class SliceProbe(TypedDict):
    """Everything the report states about one item slice."""

    n: int
    n_hops: int
    curve: list[BudgetReport]
    diagnosis: DiagnosisReport


class MultiHopProbeReport(TypedDict):
    """The probe artifact: the k curve, the per-hop control, and the named explanation."""

    lane: str
    focus_slice: str
    budgets: list[int]
    probe_depth: int
    confidence: float
    resamples: int
    seed: int
    n_items: int
    overall: SliceProbe
    slices: dict[str, SliceProbe]
    items: list[ItemProbe]


class DiagnosisCohortConversion(TypedDict):
    """Prepared-vs-raw outcomes for items sharing one RAW-query diagnosis."""

    n: int
    all_spans_before: int
    all_spans_after: int
    all_spans_gained: int
    all_spans_lost: int
    span_coverage_before: float
    span_coverage_after: float
    span_coverage_improved: int
    span_coverage_tied: int
    span_coverage_regressed: int
    newly_reachable_at_depth: int
    no_longer_reachable_at_depth: int


class QueryPrepConversion(TypedDict):
    """Per-diagnosis conversion and cost at the operating retrieval budget."""

    focus_slice: str
    n: int
    operating_budget: int
    cohorts: dict[str, DiagnosisCohortConversion]
    transitions: dict[str, dict[str, int]]


class MultiHopQueryPrepReport(TypedDict):
    """The raw and prepared probes plus their paired per-item conversion reading."""

    query_prep_steps: list[str]
    baseline: MultiHopProbeReport
    prepared: MultiHopProbeReport
    conversion: QueryPrepConversion
    endpoint: NotRequired[dict[str, str]]


def parse_budgets(spec: str) -> tuple[int, ...]:
    """Parse a `10,25,50` budget grid into sorted unique positive cutoffs."""
    budgets: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            budget = int(token)
        except ValueError:
            raise ValueError(f"retrieval budget must be an integer, got {token!r}") from None
        if budget < 1:
            raise ValueError(f"retrieval budget must be at least 1, got {budget}")
        budgets.append(budget)
    if not budgets:
        raise ValueError("no retrieval budget parsed from the grid spec")
    return tuple(sorted(dict.fromkeys(budgets)))
