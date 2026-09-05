"""The `compare-retrieval` sidecar: every lane, its paired reading, and the verdict.

This is the artifact an adoption decision is re-read from, so its shape is the whole point: a lane
row without its paired interval, or a verdict without the baseline it was drawn against, is a
number nobody can act on. Everything below mirrors the in-memory report the comparison already
builds; registering it is what makes an archived one readable by a build that has moved on.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.retrieval_graph.common import RetrievalRow
from llb.core.contracts.retrieval_graph.statistics import PairedRow, SelectionAdjustmentRow
from llb.core.contracts.retrieval_graph.stores import DuplicateCensus

RETRIEVAL_COMPARISON_SCHEMA_ID = "llb.retrieval-comparison"


class ComparisonLaneRow(RetrievalRow):
    """One lane's span metrics, with the paired reading that says whether it separates."""

    n: int = Field(ge=0)
    k: int = Field(ge=0)
    recall_at_k: float
    mrr: float
    span_char_coverage_at_k: float | None = None
    span_intact_at_k: float | None = None
    served_chars_at_k: float | None = None
    paired_vs_baseline: PairedRow | None = None


class ComparisonSliceRow(RetrievalRow):
    """One question-type slice: its item count and every lane scored on those items alone."""

    n: int = Field(ge=0)
    backends: dict[str, ComparisonLaneRow] = Field(default_factory=dict)


class ComparisonItemOutcomeRow(RetrievalRow):
    """One item's per-lane metric values, the draw every paired interval is resampled from."""

    item_id: str
    lanes: dict[str, dict[str, float]] = Field(default_factory=dict)


class ComparisonUncertaintyRow(RetrievalRow):
    """What the intervals were drawn with, so a report is reproducible from its own header."""

    baseline: str | None = None
    eligible_lanes: list[str] = Field(default_factory=list)
    resamples: int = Field(ge=0)
    confidence: float
    seed: int


class ComparisonVerdictRow(RetrievalRow):
    """Adopt-or-retain against the declared baseline, and why it says that."""

    decision: str
    lane: str | None = None
    baseline: str | None = None
    reason: str
    selection_adjustment: SelectionAdjustmentRow | None = None


class StitchCensusRow(RetrievalRow):
    """Per-query stitching accounting: how much merged, and what it cost in served characters."""

    queries: float
    parts_per_query: float
    blocks_per_query: float
    merged_per_query: float
    chars_per_query: float
    chars_delta_per_query: float
    stitch_ms_per_query: float


class StitchLaneRow(RetrievalRow):
    """One stitched twin's accounting, plus the invariance its reading depends on."""

    base: str
    census: StitchCensusRow
    recall_invariant: bool
    coverage_invariant: bool


class ValueSpreadRow(RetrievalRow):
    """Band of one number across replicate measurements, beside the value the report quotes."""

    base: float
    min: float
    max: float
    mean: float
    std: float
    half_width: float


class LaneFloorRow(RetrievalRow):
    """One lane's metric bands plus the fragility that explains how wide they are."""

    recall_at_k: ValueSpreadRow
    mrr: ValueSpreadRow
    n: int = Field(ge=0)
    fragile_items: int = Field(ge=0)
    tie_items: int = Field(ge=0)
    cut_block: float


class FloorMarginRow(RetrievalRow):
    """The reading a recommendation rests on: the top two lanes and their gap versus the floor."""

    leader: str
    runner_up: str | None = None
    delta: float
    floor: float
    clears_floor: bool
    clearance: float | None = None
    floor_multiple: float | None = None


class NoiseFloorRow(RetrievalRow):
    """Per-lane metric spread under score noise, plus the worst-lane floor per metric."""

    replicates: int = Field(ge=0)
    jitter: float
    candidates: int = Field(ge=0)
    seed: int
    lanes: dict[str, LaneFloorRow] = Field(default_factory=dict)
    unscored: list[str] = Field(default_factory=list)
    jitter_by_lane: dict[str, float] | None = None
    floor_recall_at_k: float
    floor_mrr: float
    margin: FloorMarginRow | None = None  # absent when no lane was measured


class RetrievalComparisonReport(ArtifactContract):
    """The `compare-retrieval` sidecar: every lane, its paired reading, and the verdict."""

    schema_id: Literal["llb.retrieval-comparison"]
    schema_version: Literal["1.0.0"]
    k: int = Field(ge=0)
    n: int = Field(ge=0)
    backends: dict[str, ComparisonLaneRow] = Field(default_factory=dict)
    best_recall: str | None = None
    paired_items: list[ComparisonItemOutcomeRow] = Field(default_factory=list)
    uncertainty: ComparisonUncertaintyRow
    verdict: ComparisonVerdictRow
    slices: dict[str, ComparisonSliceRow] | None = None
    duplicates: dict[str, DuplicateCensus] | None = None
    duplicates_kept: dict[str, str] | None = None
    stitching: dict[str, StitchLaneRow] | None = None
    noise_floor: NoiseFloorRow | None = None
