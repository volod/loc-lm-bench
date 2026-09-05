"""The `calibrate-fusion-routing` sidecar: every tuned policy, the frozen one, the decision."""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.retrieval_graph.common import RetrievalRow
from llb.core.contracts.retrieval_graph.statistics import BootstrapRatioRow, PairedComparisonRow

ROUTING_CALIBRATION_SCHEMA_ID = "llb.fusion-routing-calibration"


class PolicySpecRow(RetrievalRow):
    """The thresholds one routing policy is defined by."""

    label: str
    long_question_words: int = Field(ge=0)
    min_linked_entities: int = Field(ge=0)


class RouteQualityRow(RetrievalRow):
    """How often a policy routed to the graph lane when it should have, and vice versa."""

    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: BootstrapRatioRow
    recall: BootstrapRatioRow


class RouteErrorRow(RetrievalRow):
    """One misrouted item, with the signals the policy decided it on."""

    item_id: str
    predicted: str
    expected: str
    signals: list[str] = Field(default_factory=list)


class PolicyResultRow(RetrievalRow):
    """One policy's whole reading on one split."""

    policy: PolicySpecRow
    n: int = Field(ge=0)
    multi_span_n: int = Field(ge=0)
    single_span_n: int = Field(ge=0)
    graph_questions: int = Field(ge=0)
    vector_questions: int = Field(ge=0)
    route: RouteQualityRow
    route_errors: list[RouteErrorRow] = Field(default_factory=list)
    multi_span_coverage: PairedComparisonRow
    single_span_recall: PairedComparisonRow
    recommendation_gate: bool


class RoutingCalibrationReport(ArtifactContract):
    """The `calibrate-fusion-routing` sidecar: every tuned policy, the frozen one, the decision.

    The tuning/final split is what the record exists to preserve: `tuning` holds every policy that
    was searched, `frozen_policy` names the one chosen from it, and `final` is that policy alone
    scored on held-out items. A reader that cannot tell those apart cannot tell a calibrated
    result from a searched one.
    """

    schema_id: Literal["llb.fusion-routing-calibration"]
    schema_version: Literal["1.0.0"]
    k: int = Field(ge=0)
    graph_strategy: str
    graph_weight: float
    candidates: int = Field(ge=0)
    span_identity: str
    tuning_split: str
    final_split: str
    resamples: int = Field(ge=0)
    confidence: float
    seed: int
    sidecar_hidden: bool
    selection_metric: str
    tuning: dict[str, PolicyResultRow] = Field(default_factory=dict)
    frozen_policy: str
    recommended_policy: str | None = None
    final: PolicyResultRow
    decision: str
    reason: str
