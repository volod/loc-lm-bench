"""Artifact schema for held-out calibration of the sidecar-free fusion router."""

from typing_extensions import TypedDict

from llb.rag.fusion_evidence.stats import Interval, PairedComparison


class PolicySpec(TypedDict):
    label: str
    long_question_words: int
    min_linked_entities: int


class RouteQuality(TypedDict):
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: Interval
    recall: Interval


class RouteError(TypedDict):
    item_id: str
    predicted: str
    expected: str
    signals: tuple[str, ...]


class PolicyResult(TypedDict):
    policy: PolicySpec
    n: int
    multi_span_n: int
    single_span_n: int
    graph_questions: int
    vector_questions: int
    route: RouteQuality
    route_errors: list[RouteError]
    multi_span_coverage: PairedComparison
    single_span_recall: PairedComparison
    recommendation_gate: bool


class RoutingCalibrationReport(TypedDict):
    k: int
    graph_strategy: str
    graph_weight: float
    candidates: int
    span_identity: str
    tuning_split: str
    final_split: str
    resamples: int
    confidence: float
    seed: int
    sidecar_hidden: bool
    selection_metric: str
    tuning: dict[str, PolicyResult]
    frozen_policy: str
    recommended_policy: str | None
    final: PolicyResult
    decision: str
    reason: str
