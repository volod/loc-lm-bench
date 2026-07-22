"""Types and vocabulary of the retrieval-lane answer-quality comparison.

Retrieval coverage is not answer quality. The fusion-evidence lane measures whether a context
CARRIES every span a multi-hop answer needs; it cannot say whether the model then USES both. This
lane scores the same items end to end (retrieve -> generate -> score) under two retrieval lanes and
reports the objective per question-type slice, so a measured coverage gain is either confirmed as
an answer-quality gain or recorded as a retrieval-only effect.
"""

from typing import NamedTuple

from typing_extensions import TypedDict

from llb.rag.fusion_evidence.slices import SliceReport

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

# The coverage metric the retrieval-only verdict is stated on: the most SENSITIVE one every lane
# measured, falling back through the coarser ones. `span_coverage` leads because it is graded --
# on a hard multi-hop slice `all_spans_at_k` can be uniformly 0.0 for every lane (no item gets both
# hops at k), which makes the gate blind to a lane that nonetheless carried more of the evidence.
COVERAGE_PRIORITY = (METRIC_SPAN_COVERAGE, METRIC_ALL_SPANS, METRIC_RETRIEVAL_HIT)

# The slice the verdict is decided on; other question types still report as context slices.
FOCUS_SLICE = "multi-hop"

# The candidate lane's objective beats the baseline with its paired interval clear of zero.
VERDICT_ANSWER_GAIN = "answer_quality_gain"
# The candidate retrieves more evidence but does not turn it into better answers.
VERDICT_RETRIEVAL_ONLY = "retrieval_only"
# A positive objective point estimate whose paired interval still includes no difference.
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_NO_GAIN = "no_gain"
VERDICT_NO_EVIDENCE = "no_evidence"


class LaneSpec(NamedTuple):
    """One scored retrieval lane: its row label plus the retrieval knobs that define it.

    The label is the same string the fusion sweep prints (`vector`,
    `fused/global_community@0.10/d10`), so an operator can paste a sweep verdict's `best_row`
    straight into this lane's selection.
    """

    label: str
    retrieval_backend: str
    retrieval_strategy: str | None = None
    graph_weight: float | None = None
    graph_fusion_candidates: int | None = None


class LaneReport(TypedDict):
    """One scored lane: its run bundles, overall metrics, and every question-type slice."""

    label: str
    run_dirs: list[str]
    overall: SliceReport
    slices: dict[str, SliceReport]


class ItemOutcome(TypedDict):
    """Item-level paired outcome on the focus slice -- the small-n reviewer view."""

    item_id: str
    question_type: str | None
    lanes: dict[str, dict[str, float]]


class AnswerQualityVerdict(TypedDict):
    """Whether the candidate lane's retrieval gain reaches the answer."""

    focus_slice: str
    focus_n: int
    baseline: str
    best_lane: str | None
    coverage_metric: str
    decision: str
    reason: str


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
