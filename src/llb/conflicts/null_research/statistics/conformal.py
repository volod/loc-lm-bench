"""Group-split conformal tail certification against the two-way row bootstrap.

A shipped threshold is fitted ONCE and then used on every later pair, so the quantity an operator
needs is training-conditional: with the bank I actually drew, is the tail rate at this threshold
below `alpha`? A marginal guarantee ("on average over banks") does not answer that -- it is satisfied
by a threshold that overshoots half the time.

Group-split conformal answers it exactly when the units are exchangeable with the thresholded
population. Take one statistic per independent unit (its worst row), sort them, and use the k-th
largest as the threshold: the exceedance probability of that order statistic is Beta(k, n+1-k), so
`P(tail <= alpha) = P(Binomial(n, alpha) >= k)` -- a distribution-free statement needing no model of
the score at all. It also refuses to answer: when no rank reaches the requested confidence, the bank
cannot certify the tail and no threshold is returned.

Whether that beats the shipped two-way row bootstrap is an empirical question with three parts, and
this lane answers all three by simulation, so no corpus, encoder, or model is involved:

1. duplicate reuse -- many rows per unit, the shape every control bank in this project has;
2. domain shift -- calibration units drawn beside, not from, the population being thresholded;
3. sparse tail -- a tail finer than the bank can resolve.

Both methods are scored on the SAME claim: the upper bound each one publishes for the tail rate on a
fresh population. A bound that is never wrong because it is never made is reported as a refusal, not
as coverage.
"""

from dataclasses import dataclass
import math

import numpy as np

from llb.conflicts.null_research.statistics.clusters import two_way_tail_interval
from llb.conflicts.null_research.evaluation import MIN_COVERAGE_PROBABILITY
from llb.core.contracts.common import JsonObject

CONFORMAL_METHOD = "group_split_conformal_tail"
DEFAULT_UNIT_GRID = (25, 50, 100, 200)
DEFAULT_REPLICATIONS = 100
DEFAULT_ROWS_PER_UNIT = 8
DEFAULT_BOOTSTRAP_DRAWS = 100
DEFAULT_CONFIDENCE = 0.95
TEST_UNITS = 400
DOMAIN_SHIFT = 0.35
# Row noise inside a unit: small enough that rows of one unit stay near-duplicates of each other,
# which is exactly why row counts overstate the evidence a control bank carries.
WITHIN_UNIT_SCALE = 0.15


def binomial_tail(units: int, alpha: float, rank: int) -> float:
    """`P(Binomial(units, alpha) >= rank)` -- the confidence the rank-k threshold certifies."""
    return sum(
        math.comb(units, successes) * alpha**successes * (1.0 - alpha) ** (units - successes)
        for successes in range(rank, units + 1)
    )


def tolerance_rank(units: int, alpha: float, confidence: float) -> int | None:
    """The strictest order statistic whose tail stays under `alpha` with `confidence`.

    Larger ranks give a lower (more permissive) threshold, so the LARGEST admissible rank is the
    most useful one. `None` means no rank reaches the requested confidence: this many units cannot
    certify this tail, whatever their values are.
    """
    admissible = [
        rank for rank in range(1, units + 1) if binomial_tail(units, alpha, rank) >= confidence
    ]
    return max(admissible) if admissible else None


def certifiable_units(alpha: float, confidence: float) -> int:
    """The fewest independent units that can certify tail `alpha` at `confidence`, at any rank.

    The rank-1 threshold (the bank's worst unit) is the strictest available, so it decides the
    minimum: `1 - (1 - alpha) ** units >= confidence`.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("alpha and confidence must be between zero and one")
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - alpha))


@dataclass(frozen=True)
class Scenario:
    """One stress: how calibration units are drawn versus the population being thresholded."""

    name: str
    description: str
    calibration_shift: float
    alpha: float

    def draw(self, rng: np.random.Generator, units: int, rows_per_unit: int) -> np.ndarray:
        """A (units x rows) score matrix whose rows share a unit-level location."""
        centers = rng.normal(self.calibration_shift, 1.0, units)[:, None]
        return centers + rng.normal(0.0, WITHIN_UNIT_SCALE, (units, rows_per_unit))


SCENARIOS = (
    Scenario(
        name="duplicate_reference_reuse",
        description="exchangeable units, many near-duplicate rows each",
        calibration_shift=0.0,
        alpha=0.05,
    ),
    Scenario(
        name="domain_shift",
        description="calibration units drawn beside the thresholded population",
        calibration_shift=-DOMAIN_SHIFT,
        alpha=0.05,
    ),
    Scenario(
        name="sparse_tail",
        description="a tail finer than a bank of this size resolves",
        calibration_shift=0.0,
        alpha=0.005,
    ),
)


def conformal_threshold(unit_statistics: np.ndarray, alpha: float, confidence: float) -> float:
    """The certified group-split threshold, or infinity when the bank cannot certify the tail."""
    rank = tolerance_rank(len(unit_statistics), alpha, confidence)
    if rank is None:
        return float("inf")
    return float(np.sort(unit_statistics)[len(unit_statistics) - rank])


def _bootstrap_bound(matrix: np.ndarray, threshold: float, seed: int, draws: int) -> float:
    """The upper end of the two-way clustered interval -- the bound the row estimator publishes."""
    payload = two_way_tail_interval(
        (matrix >= threshold).astype("float64"),
        np.ones(matrix.shape[1], dtype="float64"),
        seed,
        draws=draws,
    )
    bounds = payload["tail_rate_two_way_95"]
    assert isinstance(bounds, list)
    return float(bounds[1])


def _replicate(
    scenario: Scenario,
    rng: np.random.Generator,
    *,
    units: int,
    rows_per_unit: int,
    confidence: float,
    seed: int,
    draws: int,
) -> tuple[bool | None, bool]:
    """One draw: is each method's published upper bound actually an upper bound?"""
    calibration = scenario.draw(rng, units, rows_per_unit)
    population = Scenario(scenario.name, "", 0.0, scenario.alpha)
    test = population.draw(rng, TEST_UNITS, rows_per_unit)
    unit_threshold = conformal_threshold(calibration.max(axis=1), scenario.alpha, confidence)
    conformal = (
        None
        if math.isinf(unit_threshold)
        else bool(float((test.max(axis=1) >= unit_threshold).mean()) <= scenario.alpha)
    )
    row_threshold = float(np.quantile(calibration.reshape(-1), 1.0 - scenario.alpha))
    row_rate = float((test.reshape(-1) >= row_threshold).mean())
    return conformal, row_rate <= _bootstrap_bound(calibration, row_threshold, seed, draws)


def _grid_point(
    scenario: Scenario,
    *,
    units: int,
    rows_per_unit: int,
    replications: int,
    confidence: float,
    seed: int,
    draws: int,
) -> JsonObject:
    rng = np.random.default_rng(seed)
    outcomes = [
        _replicate(
            scenario,
            rng,
            units=units,
            rows_per_unit=rows_per_unit,
            confidence=confidence,
            seed=seed + replication,
            draws=draws,
        )
        for replication in range(replications)
    ]
    claimed = [outcome[0] for outcome in outcomes if outcome[0] is not None]
    bootstrap = sum(outcome[1] for outcome in outcomes) / replications
    coverage = sum(claimed) / len(claimed) if claimed else 0.0
    return {
        "independent_units": units,
        "conformal_rank": tolerance_rank(units, scenario.alpha, confidence),
        "conformal_claim_rate": round(len(claimed) / replications, 6),
        "conformal_bound_coverage": round(coverage, 6),
        "row_bootstrap_bound_coverage": round(bootstrap, 6),
        "conformal_holds": bool(claimed) and coverage >= MIN_COVERAGE_PROBABILITY,
        "row_bootstrap_holds": bootstrap >= MIN_COVERAGE_PROBABILITY,
    }


def _units_required(grid: list[JsonObject], key: str) -> int | None:
    holding = [int(point["independent_units"]) for point in grid if point[key]]
    return min(holding) if holding else None


def scenario_comparison(
    scenario: Scenario,
    *,
    unit_grid: tuple[int, ...],
    rows_per_unit: int,
    replications: int,
    confidence: float,
    seed: int,
    draws: int,
) -> JsonObject:
    """Bound coverage of both estimators across the unit grid for one stress scenario."""
    grid = [
        _grid_point(
            scenario,
            units=units,
            rows_per_unit=rows_per_unit,
            replications=replications,
            confidence=confidence,
            seed=seed + position * replications,
            draws=draws,
        )
        for position, units in enumerate(unit_grid)
    ]
    conformal_units = _units_required(grid, "conformal_holds")
    bootstrap_units = _units_required(grid, "row_bootstrap_holds")
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "nominal_alpha": scenario.alpha,
        "distribution_free_units": certifiable_units(scenario.alpha, confidence),
        "grid": grid,
        "conformal_units_required": conformal_units,
        "row_bootstrap_units_required": bootstrap_units,
        "conformal_holds": conformal_units is not None,
        "row_bootstrap_holds": bootstrap_units is not None,
        "conformal_needs_no_more_units": bool(
            conformal_units is not None
            and (bootstrap_units is None or conformal_units <= bootstrap_units)
        ),
    }


def conformal_lane(
    *,
    unit_grid: tuple[int, ...] = DEFAULT_UNIT_GRID,
    rows_per_unit: int = DEFAULT_ROWS_PER_UNIT,
    replications: int = DEFAULT_REPLICATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> JsonObject:
    """Compare group-split conformal tail certification against the two-way row bootstrap."""
    scenarios = [
        scenario_comparison(
            scenario,
            unit_grid=unit_grid,
            rows_per_unit=rows_per_unit,
            replications=replications,
            confidence=confidence,
            seed=seed + position * replications * len(unit_grid),
            draws=draws,
        )
        for position, scenario in enumerate(SCENARIOS)
    ]
    exchangeable = [payload for payload in scenarios if payload["scenario"] != "domain_shift"]
    shifted = [payload for payload in scenarios if payload["scenario"] == "domain_shift"]
    gates = {
        "holds_under_exchangeable_units": all(
            bool(payload["conformal_holds"]) for payload in exchangeable
        ),
        "needs_no_more_units": all(
            bool(payload["conformal_needs_no_more_units"]) for payload in exchangeable
        ),
        "survives_domain_shift": all(bool(payload["conformal_holds"]) for payload in shifted),
    }
    return {
        "method": CONFORMAL_METHOD,
        "min_coverage_probability": MIN_COVERAGE_PROBABILITY,
        "confidence": confidence,
        "replications": replications,
        "rows_per_unit": rows_per_unit,
        "unit_grid": list(unit_grid),
        "scenarios": scenarios,
        "gates": {**gates, "accepted": all(gates.values())},
    }
