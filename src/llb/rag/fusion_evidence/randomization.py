"""Calibrated one-sided paired randomization inference.

The paired bootstrap remains the interval estimator.  A separation, however, is decided by the
sign-flip null of the observed per-item deltas: under no candidate advantage, every non-zero
delta's sign is exchangeable while its magnitude stays fixed.

Small ledgers are enumerated exactly.  Larger ledgers use a deterministic Monte Carlo draw with
the standard plus-one correction, which makes the reported p-value valid rather than occasionally
zero.  Using the same ``(discordant count, resamples, seed)`` inputs for every row also gives rows
common random signs, just as the bootstrap gives them common resample indexes.
"""

import math
from functools import lru_cache
from math import isclose
from random import Random

from typing_extensions import Literal, TypedDict

EXACT_SIGN_FLIP_LIMIT = 16
"""Largest discordant ledger enumerated in full (2**16 sign assignments)."""

DEFAULT_RANDOMIZATION_RESAMPLES = 20_000
"""Fallback draw when a caller has no bootstrap draw whose size can be reused."""

RANDOMIZATION_EXACT: Literal["exact"] = "exact"
RANDOMIZATION_MONTE_CARLO: Literal["monte_carlo"] = "monte_carlo"


class RandomizationResult(TypedDict):
    """One-sided sign-flip p-value and enough provenance to reproduce it."""

    p_value: float
    method: Literal["exact", "monte_carlo"]
    samples: int


def _at_least(value: float, observed: float) -> bool:
    """Tail comparison that treats roundoff-equivalent permutation sums as ties."""
    return value > observed or isclose(value, observed, rel_tol=1e-12, abs_tol=1e-15)


@lru_cache(maxsize=4096)
def _enumerated_tail(magnitudes: tuple[float, ...], observed: float) -> tuple[float, int]:
    samples = 2 ** len(magnitudes)
    extreme = sum(
        _at_least(_signed_sum(magnitudes, assignment), observed) for assignment in range(samples)
    )
    return extreme / samples, samples


def _exact_tail(magnitudes: list[float], observed: float) -> RandomizationResult:
    p_value, samples = _enumerated_tail(tuple(magnitudes), observed)
    return {"p_value": p_value, "method": RANDOMIZATION_EXACT, "samples": samples}


def _signed_sum(magnitudes: list[float] | tuple[float, ...], assignment: int) -> float:
    return sum(
        magnitude if assignment & (1 << index) else -magnitude
        for index, magnitude in enumerate(magnitudes)
    )


def _equal_magnitude_tail(magnitudes: list[float], observed: float) -> RandomizationResult | None:
    """Closed-form exact tail for binary/count metrics, at any discordant item count."""
    unit = magnitudes[0]
    if not all(isclose(value, unit, rel_tol=1e-12, abs_tol=1e-15) for value in magnitudes):
        return None
    discordant = len(magnitudes)
    positive_cut = math.ceil((observed / unit + discordant) / 2.0 - 1e-12)
    extreme = sum(math.comb(discordant, count) for count in range(positive_cut, discordant + 1))
    samples = 2**discordant
    result: RandomizationResult = {
        "p_value": extreme / samples,
        "method": RANDOMIZATION_EXACT,
        "samples": samples,
    }
    return result


def _monte_carlo_tail(
    magnitudes: list[float], observed: float, *, resamples: int, seed: int
) -> RandomizationResult:
    if resamples <= 0:
        raise ValueError("a Monte Carlo randomization test needs at least one resample")
    rng = Random(seed)
    extreme = 0
    width = len(magnitudes)
    for _ in range(resamples):
        randomized = _signed_sum(magnitudes, rng.getrandbits(width))
        extreme += _at_least(randomized, observed)
    # Include the observed assignment.  Besides preventing a reported zero, this correction is
    # what makes a sampled randomization p-value conservative under the null.
    return {
        "p_value": (extreme + 1) / (resamples + 1),
        "method": RANDOMIZATION_MONTE_CARLO,
        "samples": resamples,
    }


def paired_randomization(
    deltas: list[float],
    *,
    resamples: int = DEFAULT_RANDOMIZATION_RESAMPLES,
    seed: int,
    exact_limit: int = EXACT_SIGN_FLIP_LIMIT,
) -> RandomizationResult:
    """Test whether the mean paired delta is greater than zero under sign exchangeability."""
    if exact_limit < 0:
        raise ValueError("exact_limit must be non-negative")
    nonzero = [value for value in deltas if value != 0.0]
    if not nonzero:
        return {"p_value": 1.0, "method": RANDOMIZATION_EXACT, "samples": 1}
    magnitudes = [abs(value) for value in nonzero]
    observed = sum(nonzero)
    closed_form = _equal_magnitude_tail(magnitudes, observed)
    if closed_form is not None:
        return closed_form
    if len(nonzero) <= exact_limit:
        return _exact_tail(magnitudes, observed)
    return _monte_carlo_tail(magnitudes, observed, resamples=resamples, seed=seed)


def randomization_alpha(confidence: float) -> float:
    """One-sided alpha corresponding to a two-sided interval confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    return (1.0 - confidence) / 2.0


def randomization_separates(p_value: float, confidence: float) -> bool:
    """Whether a candidate-ahead randomization p-value clears the reporting convention."""
    return p_value <= randomization_alpha(confidence)


def seed_from_index_sets(index_sets: list[list[int]]) -> int:
    """Stable sign-draw seed derived from the bootstrap draw a comparison already receives."""
    # FNV-1a over a bounded prefix: deterministic across Python processes and distinct for
    # configured bootstrap seeds without rescanning n * resamples for every metric and grid row.
    value = 14_695_981_039_346_656_037
    prime = 1_099_511_628_211
    value = ((value ^ len(index_sets)) * prime) & 0xFFFFFFFFFFFFFFFF
    for indexes in index_sets[:4]:
        value = ((value ^ len(indexes)) * prime) & 0xFFFFFFFFFFFFFFFF
        for index in indexes[:16]:
            value = ((value ^ index) * prime) & 0xFFFFFFFFFFFFFFFF
    return value
