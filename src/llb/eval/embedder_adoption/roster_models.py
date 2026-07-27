"""Types and declared model properties for adoption-roster comparisons."""

from typing import TYPE_CHECKING

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from llb.rag.fusion_evidence.stability import ReadingStability as RowStability

PROPERTY_PARAMS = "params_b"
PROPERTY_FAMILY = "family"
NUMERIC_PROPERTIES = (PROPERTY_PARAMS,)
CATEGORICAL_PROPERTIES = (PROPERTY_FAMILY,)
PROPERTIES = NUMERIC_PROPERTIES + CATEGORICAL_PROPERTIES

DECISION_PROPERTY_PREDICTS = "property_predicts"
DECISION_NO_PROPERTY_PREDICTS = "no_property_predicts"
DECISION_INSUFFICIENT_VARIATION = "insufficient_variation"


class ModelProfile(TypedDict):
    params_b: NotRequired[float]
    family: NotRequired[str]


class RosterCell(TypedDict):
    label: str
    readings: dict[str, str]
    unanimous: bool
    answer_models: list[str]
    stability: NotRequired[dict[str, "RowStability"]]


class PropertySeparation(TypedDict):
    property: str
    separates: bool
    chance_probability: NotRequired[float]
    missing: list[str]
    reason: str


class RosterVerdict(TypedDict):
    decision: str
    focus_cell: str
    answer_models: list[str]
    other_models: list[str]
    separations: list[PropertySeparation]
    reason: str


class RosterReport(TypedDict):
    models: list[str]
    profiles: dict[str, ModelProfile]
    baseline: str
    candidate: str
    n: int
    focus_cell: str
    verdicts: dict[str, str]
    cells: list[RosterCell]
    verdict: RosterVerdict
    metadata: NotRequired[dict[str, object]]
