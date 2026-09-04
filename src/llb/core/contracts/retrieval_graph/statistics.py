"""Paired-statistics rows shared by the retrieval comparison and routing calibration records.

Both sidecars answer the same question -- does this lane separate from its baseline, and how
sure is the reading -- so both state it in the same rows. Keeping the shapes in one module is
what makes a comparison interval and a calibration interval readable side by side.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.retrieval_graph.common import RetrievalRow


class IntervalRow(RetrievalRow):
    """A point estimate with its percentile-bootstrap confidence bounds."""

    mean: float
    lo: float
    hi: float


class ReadingStabilityRow(RetrievalRow):
    """One row's reading plus how close it sits to the cut that produced it.

    `looser_reading` / `tighter_reading` are what the neighbouring confidence conventions read, so
    a `borderline` row is one whose reading is the convention's rather than the data's; `side`
    says which way. `discordant` and `pairs` are absent on an artifact recorded before the
    evidence gate existed.
    """

    reading: str
    p_positive: float
    randomization_p: float | None = None
    randomization_method: str | None = None
    randomization_samples: int | None = None
    looser_reading: str
    tighter_reading: str
    borderline: bool
    side: str | None = None
    discordant: int | None = None
    pairs: int | None = None


class BootstrapRatioRow(IntervalRow):
    """A count ratio whose lower-bound reading is qualified from the same bootstrap draw."""

    stability: ReadingStabilityRow | None = None


class PairedComparisonRow(RetrievalRow):
    """A candidate-minus-baseline delta plus the item-level win/loss/tie ledger."""

    delta: IntervalRow
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)
    sign_test_p: float
    randomization_p: float | None = None
    randomization_method: str | None = None
    randomization_samples: int | None = None
    stability: ReadingStabilityRow | None = None


class SelectionPValueRow(RetrievalRow):
    """Observed statistic plus its per-test and family-adjusted randomization p-values."""

    statistic: float
    unadjusted_p: float
    adjusted_p: float


class SelectionAdjustmentRow(RetrievalRow):
    """Reproducible Westfall-Young reading over one declared hypothesis family."""

    method: Literal["westfall_young_step_down_max_t"]
    statistic: Literal["studentized_mean"]
    randomization_method: Literal["exact", "monte_carlo"]
    samples: int = Field(ge=0)
    seed: int
    items: int = Field(ge=0)
    family_size: int = Field(ge=0)
    p_values: dict[str, SelectionPValueRow] = Field(default_factory=dict)


class PairedRow(RetrievalRow):
    """One lane's paired delta against the baseline, keyed by metric."""

    baseline: str
    metrics: dict[str, PairedComparisonRow] = Field(default_factory=dict)
