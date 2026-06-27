"""Statistical helpers for the guarded category composite."""

import random
from collections.abc import Mapping, Sequence

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0
ROUND_DIGITS = 4


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))
    return sorted_values[index]


def bootstrap_weighted_mean_ci(
    weighted_series: Sequence[tuple[str, Sequence[float]]],
    weights: Mapping[str, float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float] | None:
    """Bootstrap the weighted mean of per-category case-score series."""
    if any(len(values) < 2 for _tier, values in weighted_series):
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for tier, values in weighted_series:
            sampled = [values[rng.randrange(len(values))] for _ in range(len(values))]
            total += weights[tier] * mean(sampled)
        means.append(total)
    means.sort()
    return (
        round(percentile(means, 0.025), ROUND_DIGITS),
        round(percentile(means, 0.975), ROUND_DIGITS),
    )
