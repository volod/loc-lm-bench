"""Types and vocabulary of the retrieval-lane answer-quality comparison.

Retrieval coverage is not answer quality. The fusion-evidence lane measures whether a context
CARRIES every span a multi-hop answer needs; it cannot say whether the model then USES both. This
lane scores the same items end to end (retrieve -> generate -> score) under two retrieval lanes and
reports the objective per question-type slice, so a measured coverage gain is either confirmed as
an answer-quality gain or recorded as a retrieval-only effect.
"""

from typing import NamedTuple

from typing_extensions import NotRequired, TypedDict

from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion.spans import DEFAULT_SPAN_IDENTITY, SPAN_MERGE_MIN_RATIO

# Per-case columns compared between lanes, all present on every `scores.jsonl` row.
# `objective_score` is the decision metric; `retrieval_hit` is the any-span coverage signal that
# makes a retrieval-only effect visible; `token_f1` is the graded companion of the objective.
METRIC_OBJECTIVE = "objective_score"
METRIC_RETRIEVAL_HIT = "retrieval_hit"
METRIC_TOKEN_F1 = "token_f1"
BASE_METRICS = (METRIC_OBJECTIVE, METRIC_TOKEN_F1, METRIC_RETRIEVAL_HIT)

# Multi-span coverage, recomputed from the run bundle's retrieval sidecar (`coverage.py`) and
# reported only when every lane measured it. `retrieval_hit` alone cannot see a multi-hop coverage
# gain, since it credits an item that retrieved just one of its hops.
METRIC_ALL_SPANS = "all_spans_at_k"
METRIC_SPAN_COVERAGE = "span_coverage"
COVERAGE_METRICS = (METRIC_ALL_SPANS, METRIC_SPAN_COVERAGE)

# Answer-side gold-span coverage, written into every case row by the scorer
# (`llb.scoring.answer_spans`) rather than recomputed here. It answers the question `objective_score`
# cannot: a two-hop answer that states one fact fluently and omits the other earns roughly the same
# token F1 as a vague answer touching both, so the objective alone can never say whether the model
# USED both hops. Reported only when every lane measured it, which a bundle recorded before the
# metric existed did not.
METRIC_ANSWER_SPAN_COVERAGE = "answer_span_coverage"
METRIC_ANSWER_ALL_SPANS = "answer_all_spans"
ANSWER_COVERAGE_METRICS = (METRIC_ANSWER_SPAN_COVERAGE, METRIC_ANSWER_ALL_SPANS)

# The coverage metric the retrieval-only verdict is stated on: the most SENSITIVE one every lane
# measured, falling back through the coarser ones. `span_coverage` leads because it is graded --
# on a hard multi-hop slice `all_spans_at_k` can be uniformly 0.0 for every lane (no item gets both
# hops at k), which makes the gate blind to a lane that nonetheless carried more of the evidence.
COVERAGE_PRIORITY = (METRIC_SPAN_COVERAGE, METRIC_ALL_SPANS, METRIC_RETRIEVAL_HIT)

# The context BILL of the scored retrieval, in two units. `context_chars` is recomputed from the
# same sidecar as the coverage columns: the characters the lane laid into the prompt.
# `prompt_tokens` is what the backend actually consumed, present only when it reported it, and is
# the only column that can show a context SILENTLY TRUNCATED to the served window. A budget
# comparison is unreadable without them -- five times the chunks is five times the context, and a
# coverage gain bought at that price is a different result from the same gain bought for free.
# Both are reported, neither is decided on.
METRIC_CONTEXT_CHARS = "context_chars"
METRIC_PROMPT_TOKENS = "prompt_tokens"
# What prompt-side table-header restoration added on top of the retrieved context, in characters
# (table-header-context-restoration). It is 0.0 on a lane with the step off, so an off/on pair
# reports the price of the step directly; `prompt_tokens` is the same price in the model's own
# units. Recorded on every case that RETRIEVED, so the column is present in both lanes or neither.
METRIC_TABLE_HEADER_CHARS = "table_header_chars"
CONTEXT_METRICS = (METRIC_CONTEXT_CHARS, METRIC_PROMPT_TOKENS, METRIC_TABLE_HEADER_CHARS)

# The slice the verdict is decided on; other question types still report as context slices.
FOCUS_SLICE = "multi-hop"

# What the compared items were grounded on. A comparison scored on a DRAFTED ledger records
# `drafted` in every artifact, and each of its run bundles records `item_grounding: drafted` in its
# own manifest; a verified bundle records nothing, which is what a re-render checks a recorded
# bundle set against.
GROUNDING_VERIFIED = "verified"
GROUNDING_DRAFTED = "drafted"

# The candidate lane's calibrated objective test separates from the baseline.
VERDICT_ANSWER_GAIN = "answer_quality_gain"
# The candidate retrieves more evidence but does not turn it into better answers.
VERDICT_RETRIEVAL_ONLY = "retrieval_only"
# A positive objective point estimate whose calibrated paired test does not separate.
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_NO_GAIN = "no_gain"
VERDICT_NO_EVIDENCE = "no_evidence"

# Headline of a budget sweep, decided from the per-row readings and stated in the operator's own
# terms: the diagnosed budget's retrieval gain either reaches the answers, stops at retrieval, or
# is not there at all. A cost on any non-focus slice is a QUALIFIER on whichever outcome fired --
# never a fourth outcome, because "it converted and it cost the factoid slice" is one result.
CONVERSION_CONVERTED = "converted"
CONVERSION_STALLED = "stalled"


class LaneSpec(NamedTuple):
    """One scored retrieval lane: its row label plus the retrieval knobs that define it.

    The label is the same string the fusion sweep prints (`vector`,
    `fused/global_community@0.10/d10`), so an operator can paste a sweep verdict's `best_row`
    straight into this lane's selection. A lane scored at a non-shipped retrieval budget carries
    that budget in `top_k` and the matching `#k<budget>` suffix in its label
    (`llb.eval.answer_quality.budgets`); `None` leaves the config's own `top_k` alone.
    """

    label: str
    retrieval_backend: str
    retrieval_strategy: str | None = None
    graph_weight: float | None = None
    graph_fusion_candidates: int | None = None
    graph_fusion_span_identity: str = DEFAULT_SPAN_IDENTITY
    graph_fusion_span_merge_ratio: float = SPAN_MERGE_MIN_RATIO
    graph_fusion_router: str = "fixed"
    top_k: int | None = None
    # Prompt-side only (`llb.eval.answer_quality.table_headers`), carried in the label as
    # `+headers`. It is deliberately NOT a retrieval knob: the two lanes it distinguishes retrieve
    # identically, so any delta between them is an answer-quality delta by construction.
    restore_table_headers: bool = False


# A case row's terminal status; anything else is a case the model never answered.
STATUS_OK = "ok"


class LaneReport(TypedDict):
    """One scored lane: its run bundles, overall metrics, and every question-type slice."""

    label: str
    run_dirs: list[str]
    overall: SliceReport
    slices: dict[str, SliceReport]
    # Cases this lane did not answer (timeout, backend error, refusal). They score zero like a
    # wrong answer and are indistinguishable from one in every metric column, so the count is
    # carried explicitly: a lane whose context grew until requests timed out would otherwise read
    # as a lane whose ANSWERS got worse, which is a different finding entirely.
    not_ok: int


class CrossReading(LaneReport):
    """One lane read against ANOTHER scored lane instead of against the report baseline.

    Every candidate is normally paired against one baseline, which is the right shape for "is this
    row better than the shipped one". A budget sweep asks a second question the baseline cannot
    answer -- is the SAME row better at a larger `top_k` than at its own shipped one -- and that is
    a pairing between two candidates. The block is the LaneReport shape on purpose, so the same
    `judge_lane` decides it and no second verdict vocabulary appears in the artifact.
    """

    base_lane: str


class ItemOutcome(TypedDict):
    """Item-level paired outcome on the focus slice -- the small-n reviewer view."""

    item_id: str
    question_type: str | None
    lanes: dict[str, dict[str, float]]


class LaneDecision(TypedDict):
    """One candidate lane's own decision against the baseline."""

    decision: str
    reason: str


class AnswerQualityVerdict(TypedDict):
    """Whether the candidate lane's retrieval gain reaches the answer.

    `decision` / `reason` / `best_lane` describe the WINNING candidate; `lane_decisions` carries
    the same judgment for every candidate, because a comparison of three or more lanes has a
    result per lane and reporting only the winner would drop the rest.
    """

    focus_slice: str
    focus_n: int
    baseline: str
    best_lane: str | None
    coverage_metric: str
    decision: str
    reason: str
    lane_decisions: dict[str, LaneDecision]


class RowConversion(TypedDict):
    """Whether ONE retrieval row's extra budget reached the answers, and what it cost.

    `decision` reuses the lane vocabulary above, because the question is the same one (`did the
    retrieval gain reach the answer`) asked with the row's own smaller budget as the baseline.
    `cost_slices` names the question types whose objective the extra context measurably LOWERED --
    the failure mode the `overlap` span identity already produced on the factoid slice.
    """

    row: str
    lane: str
    base_lane: str
    budget: int
    base_budget: int
    decision: str
    reason: str
    cost_slices: list[str]


class BudgetConversion(TypedDict):
    """The budget sweep's headline: did the diagnosed budget buy answers or only retrieval?"""

    budgets: list[int]
    focus_slice: str
    coverage_metric: str
    decision: str
    reason: str
    rows: list[RowConversion]


class AnswerQualityReport(TypedDict):
    """The full lane artifact: per-lane slices, the focus-slice item ledger, and the verdict."""

    n: int
    baseline: str
    focus_slice: str
    metrics: list[str]
    resamples: int
    confidence: float
    seed: int
    item_ids: list[str]
    lanes: dict[str, LaneReport]
    focus_items: list[ItemOutcome]
    verdict: AnswerQualityVerdict
    # Present only on a budget sweep: the same rows read against themselves at a smaller `top_k`,
    # plus the conversion verdict decided from those readings.
    cross_readings: NotRequired[dict[str, CrossReading]]
    budget_conversion: NotRequired[BudgetConversion]
