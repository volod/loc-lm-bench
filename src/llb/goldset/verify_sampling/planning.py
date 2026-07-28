"""Experiment-derived sample-size planning for human verification."""

import math
from statistics import NormalDist
from typing import TypedDict

DEFAULT_SAMPLE_CONFIDENCE = 0.95
DEFAULT_SAMPLE_PRECISION = 0.10
DEFAULT_EXPECTED_REJECT_RATE = 0.50
SAMPLE_GATE_ID = "verification-sample-precision"
SAMPLE_GATE_METHOD = "normal-worst-case-finite-population"


class SampleSizePlan(TypedDict):
    gate_id: str
    classification: str
    method: str
    assumptions: dict[str, float | int]
    derived_target: int
    operator_override: int | None
    selected_target: int
    override_meets_derived_target: bool


def required_sample_size(
    population: int,
    *,
    confidence: float = DEFAULT_SAMPLE_CONFIDENCE,
    precision: float = DEFAULT_SAMPLE_PRECISION,
    expected_reject_rate: float = DEFAULT_EXPECTED_REJECT_RATE,
) -> int:
    """Price a proportion estimate, including finite-population correction."""
    if population < 1:
        raise ValueError("verification population must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("sample confidence must be between zero and one")
    if not 0.0 < precision < 1.0:
        raise ValueError("sample precision must be between zero and one")
    if not 0.0 < expected_reject_rate < 1.0:
        raise ValueError("expected reject rate must be between zero and one")
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    infinite_n = critical**2 * expected_reject_rate * (1.0 - expected_reject_rate) / precision**2
    corrected = population * infinite_n / (population + infinite_n - 1.0)
    return min(population, max(1, math.ceil(corrected)))


def sample_size_plan(
    population: int,
    strata: int,
    *,
    requested_size: int | None,
    confidence: float = DEFAULT_SAMPLE_CONFIDENCE,
    precision: float = DEFAULT_SAMPLE_PRECISION,
    expected_reject_rate: float = DEFAULT_EXPECTED_REJECT_RATE,
) -> SampleSizePlan:
    """Return the declared assumptions, derived target, and selected sample size."""
    if strata < 1 or strata > population:
        raise ValueError("verification strata must be between one and the population")
    derived = max(
        strata,
        required_sample_size(
            population,
            confidence=confidence,
            precision=precision,
            expected_reject_rate=expected_reject_rate,
        ),
    )
    if requested_size is not None and requested_size < 1:
        raise ValueError("verification sample override must be positive")
    selected = min(population, requested_size if requested_size is not None else derived)
    return {
        "gate_id": SAMPLE_GATE_ID,
        "classification": "inferential_gate",
        "method": SAMPLE_GATE_METHOD,
        "assumptions": {
            "confidence": confidence,
            "precision": precision,
            "expected_reject_rate": expected_reject_rate,
            "finite_population": population,
            "strata_floor": strata,
        },
        "derived_target": derived,
        "operator_override": requested_size,
        "selected_target": selected,
        "override_meets_derived_target": requested_size is None or selected >= derived,
    }
