"""Family-wise inference for verdicts that select the strongest row from a grid.

Each hypothesis is an aligned candidate-minus-baseline item vector.  One joint sign-flip draw is
applied across the whole family, preserving the observed correlation between rows.  The resulting
studentized max statistics feed the Westfall-Young step-down procedure.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal, TypedDict

from llb.rag.fusion_evidence.randomization import randomization_separates

DEFAULT_SELECTION_RESAMPLES = 20_000
SELECTION_EXACT_LIMIT = 16
SELECTION_BATCH_SIZE = 2_000

SELECTION_METHOD: Literal["westfall_young_step_down_max_t"] = "westfall_young_step_down_max_t"
STATISTIC: Literal["studentized_mean"] = "studentized_mean"


class SelectionPValue(TypedDict):
    """Observed statistic plus its per-test and family-adjusted randomization p-values."""

    statistic: float
    unadjusted_p: float
    adjusted_p: float


class SelectionAdjustment(TypedDict):
    """Reproducible Westfall-Young reading over one declared hypothesis family."""

    method: Literal["westfall_young_step_down_max_t"]
    statistic: Literal["studentized_mean"]
    randomization_method: Literal["exact", "monte_carlo"]
    samples: int
    seed: int
    items: int
    family_size: int
    p_values: dict[str, SelectionPValue]


def _statistics(
    sums: NDArray[np.float64], sum_squares: NDArray[np.float64], n: int
) -> NDArray[np.float64]:
    """Studentized means from signed sums; unanimous non-zero vectors map to signed infinity."""
    centered = np.maximum(sum_squares - (sums * sums / n), 0.0)
    denominator = np.sqrt(centered * n / max(n - 1, 1))
    result = np.zeros_like(sums, dtype=np.float64)
    np.divide(sums, denominator, out=result, where=denominator > 1e-15)
    constant = denominator <= 1e-15
    result[constant & (sums > 0.0)] = np.inf
    result[constant & (sums < 0.0)] = -np.inf
    return result


def _exact_signs(start: int, stop: int, n: int) -> NDArray[np.float64]:
    assignments = np.arange(start, stop, dtype=np.uint64)
    bits = (assignments[:, None] >> np.arange(n, dtype=np.uint64)) & 1
    return bits.astype(np.float64) * 2.0 - 1.0


def _sampled_signs(rng: np.random.Generator, count: int, n: int) -> NDArray[np.float64]:
    bits = rng.integers(0, 2, size=(count, n), dtype=np.int8)
    return bits.astype(np.float64) * 2.0 - 1.0


def _tail_count(null: NDArray[np.float64], observed: float) -> int:
    if math.isinf(observed):
        return int(np.count_nonzero(null >= observed))
    tolerance = abs(observed) * 1e-12 + 1e-15
    return int(np.count_nonzero(null >= observed - tolerance))


def _aligned_matrix(hypotheses: Mapping[str, Sequence[float]], keys: list[str]) -> np.ndarray:
    """The family as one finite matrix over the SAME items, which is what makes max-T valid."""
    lengths = {len(hypotheses[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError("selection hypotheses must use the same aligned items")
    matrix = np.asarray([hypotheses[key] for key in keys], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("selection hypotheses must contain only finite deltas")
    return matrix


def _tail_counts(
    matrix: np.ndarray,
    order: list[int],
    observed: np.ndarray,
    *,
    exact: bool,
    samples: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk the randomization distribution and count, per member, how often the null was as extreme.

    Two counts, because the family needs both: the member's OWN tail (its unadjusted p) and the
    step-down suffix maximum (its adjusted p).
    """
    n = matrix.shape[1]
    sum_squares = np.square(matrix).sum(axis=1)
    ordered_observed = observed[order]
    raw_extreme = np.zeros(len(order), dtype=np.int64)
    step_extreme = np.zeros(len(order), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        signs = _exact_signs(start, start + count, n) if exact else _sampled_signs(rng, count, n)
        null_sums = matrix @ signs.T
        null_statistics = _statistics(null_sums, sum_squares[:, None], n) if n else null_sums
        ordered_null = null_statistics[order]
        suffix_max = np.maximum.accumulate(ordered_null[::-1], axis=0)[::-1]
        for position, index in enumerate(order):
            raw_extreme[index] += _tail_count(null_statistics[index], observed[index])
            step_extreme[position] += _tail_count(suffix_max[position], ordered_observed[position])
    return raw_extreme, step_extreme


def _step_down_adjusted(
    step_extreme: np.ndarray, order: list[int], *, correction: int, denominator: int
) -> dict[int, float]:
    """Enforce monotonicity down the ordered family: an adjusted p never drops below the one above."""
    adjusted_ordered: list[float] = []
    running = 0.0
    for extreme in step_extreme:
        running = max(running, (int(extreme) + correction) / denominator)
        adjusted_ordered.append(running)
    return {index: adjusted_ordered[position] for position, index in enumerate(order)}


def selection_adjustment(
    hypotheses: Mapping[str, Sequence[float]],
    *,
    resamples: int = DEFAULT_SELECTION_RESAMPLES,
    seed: int,
    exact_limit: int = SELECTION_EXACT_LIMIT,
    batch_size: int = SELECTION_BATCH_SIZE,
) -> SelectionAdjustment:
    """Compute step-down max-T adjusted p-values over aligned one-sided hypotheses."""
    if not hypotheses:
        raise ValueError("selection adjustment needs at least one hypothesis")
    if exact_limit < 0 or batch_size <= 0:
        raise ValueError("exact_limit must be non-negative and batch_size must be positive")
    keys = list(hypotheses)
    matrix = _aligned_matrix(hypotheses, keys)
    original_n = matrix.shape[1]
    # Items no hypothesis moved carry no sign information, so they leave the randomization.
    active = np.any(matrix != 0.0, axis=0) if original_n else np.asarray([], dtype=bool)
    matrix = matrix[:, active]
    n = matrix.shape[1]
    observed = (
        _statistics(matrix.sum(axis=1), np.square(matrix).sum(axis=1), n)
        if n
        else np.zeros(len(keys))
    )
    order = sorted(range(len(keys)), key=lambda index: (-observed[index], keys[index]))
    exact = n <= exact_limit
    samples = 2**n if exact else resamples
    if samples <= 0:
        raise ValueError("a Monte Carlo selection adjustment needs at least one resample")
    raw_extreme, step_extreme = _tail_counts(
        matrix,
        order,
        observed,
        exact=exact,
        samples=samples,
        batch_size=batch_size,
        seed=seed,
    )
    correction = 0 if exact else 1
    denominator = samples + correction
    adjusted = _step_down_adjusted(
        step_extreme, order, correction=correction, denominator=denominator
    )
    return {
        "method": SELECTION_METHOD,
        "statistic": STATISTIC,
        "randomization_method": "exact" if exact else "monte_carlo",
        "samples": samples,
        "seed": seed,
        "items": original_n,
        "family_size": len(keys),
        "p_values": {
            key: {
                "statistic": float(observed[index]),
                "unadjusted_p": (int(raw_extreme[index]) + correction) / denominator,
                "adjusted_p": adjusted[index],
            }
            for index, key in enumerate(keys)
        },
    }


def selection_separates(adjustment: SelectionAdjustment, key: str, confidence: float) -> bool:
    """Whether one member survives its declared family at the reporting convention."""
    return randomization_separates(adjustment["p_values"][key]["adjusted_p"], confidence)
