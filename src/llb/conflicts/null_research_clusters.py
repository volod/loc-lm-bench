"""Two-way clustered resampling for quantities whose rows share units on both sides.

A control row pairs one source chunk with one reference claim; a candidate row pairs one left chunk
with one right chunk. In both cases a single unit appears in many rows, so treating rows as
independent draws understates uncertainty. Both estimators here resample the two unit sets
INDEPENDENTLY and recompute the statistic, which is what makes the resulting interval honest about
how few distinct claims the evidence really rests on.
"""

import numpy as np

from llb.core.contracts.common import JsonObject

BOOTSTRAP_DRAWS = 200
PRECISION_BOOTSTRAP_DRAWS = 500
PRECISION_LCB_QUANTILE = 0.05


def two_way_tail_interval(
    exceedances: np.ndarray,
    weights: np.ndarray,
    seed: int,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> JsonObject:
    """Percentile interval for a weighted exceedance rate over a source-by-reference matrix."""
    rows, columns = exceedances.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype="float64")
    for draw in range(draws):
        row_index = rng.integers(0, rows, rows)
        column_index = rng.integers(0, columns, columns)
        resampled_weights = weights[column_index]
        block = exceedances[row_index][:, column_index]
        estimates[draw] = float(
            (block @ resampled_weights).sum() / (rows * resampled_weights.sum())
        )
    lower, upper = (float(value) for value in np.quantile(estimates, [0.025, 0.975]))
    return {
        "draws": draws,
        "clusters": ["source_chunk", "reference_claim"],
        "tail_rate_two_way_95": [round(lower, 9), round(upper, 9)],
        "tail_rate_two_way_width": round(upper - lower, 9),
    }


def two_way_proportion_bound(
    flags: list[bool],
    left_keys: list[str],
    right_keys: list[str],
    *,
    seed: int,
    draws: int = PRECISION_BOOTSTRAP_DRAWS,
) -> float:
    """Lower confidence bound for a proportion whose rows cluster on a left and a right key."""
    if not flags:
        return 0.0
    left_unique = sorted(set(left_keys))
    right_unique = sorted(set(right_keys))
    left_index = np.asarray([left_unique.index(key) for key in left_keys])
    right_index = np.asarray([right_unique.index(key) for key in right_keys])
    values = np.asarray(flags, dtype="float64")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype="float64")
    for draw in range(draws):
        left_counts = np.bincount(
            rng.integers(0, len(left_unique), len(left_unique)), minlength=len(left_unique)
        )
        right_counts = np.bincount(
            rng.integers(0, len(right_unique), len(right_unique)), minlength=len(right_unique)
        )
        weights = left_counts[left_index] * right_counts[right_index]
        total = float(weights.sum())
        estimates[draw] = float((weights * values).sum() / total) if total else 0.0
    return float(np.quantile(estimates, PRECISION_LCB_QUANTILE))
