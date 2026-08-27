"""What the confirmation run commits to BEFORE it starts: effect, power, and when to stop.

A bounded acceptance run validates that the roster path works. It cannot support a default-model
adoption decision, because nothing in it fixes -- in advance -- how big a gain would matter, how
many items could resolve one, or when searching harder stops changing the answer. Those three are
declared here and never re-chosen once measurement begins:

- the **minimum detectable objective gain**, the smallest quality delta an operator would swap a
  default model for;
- the **tuning-screen size**, DERIVED from that gain and an earlier run's paired variance through
  the shared paired-power contract (`llb.rag.fusion_evidence.power`) rather than picked;
- the **ranking-stability criterion and trial budget**, the two ways the multi-objective search is
  allowed to end.

Everything here is pure: the input is a list of paired per-case deltas, so the declaration is
unit-tested with plain vectors and persisted before the first cell runs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.rag.fusion_evidence.power import (
    DEFAULT_TARGET_POWER,
    PowerAnalysis,
    evidence_floor_n,
    plan_from_deltas,
    required_sample_size,
    sample_sd,
)
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE

LONG_RUN_METHOD = "joint-search-long-run"

# The smallest objective gain worth swapping a shipped default model for. Declared, never fitted.
DEFAULT_MINIMUM_DETECTABLE_GAIN = 0.05

# Multi-objective trials are spent in equal BLOCKS across every finalist, so a ranking read after a
# block compares survivors that had the same search budget.
DEFAULT_TRIAL_BLOCK = 5
DEFAULT_TRIAL_BUDGET = 30

# Consecutive block transitions whose ranking must hold before the search is called settled, and
# the pairwise rank agreement a transition needs to count as one. 1.0 means "identical order".
DEFAULT_STABILITY_BLOCKS = 2
DEFAULT_STABILITY_AGREEMENT = 1.0

# Which floor set the screen size: the power arithmetic, or the tuning split simply not having
# that many items. The second is a REAL limit on the run and is reported, never rounded away.
BINDING_POWER = "power"
BINDING_AVAILABLE = "tuning-split-exhausted"


@dataclass(frozen=True)
class ScreenSizing:
    """The derived tuning-screen case cap, and which floor bound it."""

    required_n: int
    applied_n: int
    available_n: int
    binding: str

    @property
    def satisfied(self) -> bool:
        """True when the screen actually reaches the size the declared power asked for."""
        return self.applied_n >= self.required_n

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_n": self.required_n,
            "applied_n": self.applied_n,
            "available_n": self.available_n,
            "binding": self.binding,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True)
class LongRunPlan:
    """The full predeclaration; persisted before the first screen cell runs."""

    minimum_detectable_gain: float
    target_power: float
    confidence: float
    stability_blocks: int
    stability_agreement: float
    trial_budget: int
    trial_block: int
    screen: ScreenSizing
    power: PowerAnalysis

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": LONG_RUN_METHOD,
            "minimum_detectable_gain": self.minimum_detectable_gain,
            "target_power": self.target_power,
            "confidence": self.confidence,
            "alpha": round(1.0 - self.confidence, 12),
            "stability_blocks": self.stability_blocks,
            "stability_agreement": self.stability_agreement,
            "trial_budget": self.trial_budget,
            "trial_block": self.trial_block,
            "screen": self.screen.to_dict(),
            "power": dict(self.power),
            "stopping_rule": self.stopping_rule,
        }

    @property
    def stopping_rule(self) -> str:
        """The one sentence the trial loop is actually allowed to end on."""
        return (
            f"stop when the finalist ranking survives {self.stability_blocks} consecutive block "
            f"transitions at pairwise rank agreement >= {self.stability_agreement:.2f} with an "
            f"unchanged leader, or when {self.trial_budget} trials per finalist are spent"
        )


def screen_sizing(
    deltas: list[float],
    *,
    minimum_detectable_gain: float,
    target_power: float,
    confidence: float,
    available_n: int,
) -> ScreenSizing:
    """Price the tuning-screen case cap from an earlier run's paired variance.

    Both floors the shared contract knows apply: the normal-approximation variance floor and the
    discordance floor the exact sign test needs. The screen cannot exceed the tuning split, so a
    split too small to carry the declared power is reported as such instead of being silently met.
    """
    if available_n < 1:
        raise ValueError("the tuning split must hold at least one item")
    sd = sample_sd(deltas)
    alpha = round(1.0 - confidence, 12)
    variance_n = required_sample_size(
        sd, minimum_detectable_gain, alpha=alpha, target_power=target_power
    )
    evidence_n = evidence_floor_n(deltas, confidence)
    required = max(variance_n, evidence_n or variance_n)
    applied = min(required, available_n)
    return ScreenSizing(
        required_n=required,
        applied_n=applied,
        available_n=available_n,
        binding=BINDING_POWER if applied >= required else BINDING_AVAILABLE,
    )


def declare_plan(
    reference_artifact: Path,
    deltas: list[float],
    *,
    minimum_detectable_gain: float = DEFAULT_MINIMUM_DETECTABLE_GAIN,
    target_power: float = DEFAULT_TARGET_POWER,
    confidence: float = DEFAULT_CONFIDENCE,
    available_n: int,
    trial_budget: int = DEFAULT_TRIAL_BUDGET,
    trial_block: int = DEFAULT_TRIAL_BLOCK,
    stability_blocks: int = DEFAULT_STABILITY_BLOCKS,
    stability_agreement: float = DEFAULT_STABILITY_AGREEMENT,
    selector: dict[str, str],
) -> LongRunPlan:
    """Build the complete predeclaration from an earlier paired item ledger."""
    if trial_block < 1 or trial_budget < trial_block:
        raise ValueError("the trial budget must be at least one full block")
    if stability_blocks < 1:
        raise ValueError("the stability rule needs at least one block transition")
    if not 0.0 <= stability_agreement <= 1.0:
        raise ValueError("rank agreement is a share in [0, 1]")
    sizing = screen_sizing(
        deltas,
        minimum_detectable_gain=minimum_detectable_gain,
        target_power=target_power,
        confidence=confidence,
        available_n=available_n,
    )
    power = plan_from_deltas(
        reference_artifact,
        deltas,
        minimum_detectable_delta=minimum_detectable_gain,
        target_power=target_power,
        confidence=confidence,
        planned_n=sizing.applied_n,
        selector=selector,
    )
    return LongRunPlan(
        minimum_detectable_gain=minimum_detectable_gain,
        target_power=target_power,
        confidence=confidence,
        stability_blocks=stability_blocks,
        stability_agreement=stability_agreement,
        trial_budget=trial_budget,
        trial_block=trial_block,
        screen=sizing,
        power=power,
    )


__all__ = [
    "BINDING_AVAILABLE",
    "BINDING_POWER",
    "DEFAULT_MINIMUM_DETECTABLE_GAIN",
    "DEFAULT_STABILITY_AGREEMENT",
    "DEFAULT_STABILITY_BLOCKS",
    "DEFAULT_TRIAL_BLOCK",
    "DEFAULT_TRIAL_BUDGET",
    "LONG_RUN_METHOD",
    "LongRunPlan",
    "ScreenSizing",
    "declare_plan",
    "screen_sizing",
]
