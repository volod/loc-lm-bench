"""Calibrate the semantic cosine threshold against a corpus's OWN similarity distribution.

A fixed cosine cutoff is not portable across corpora. Measured on real multilingual-E5 vectors,
two completely unrelated chunks already score 0.83 in the raw space; even after mean-centering,
the useful operating point on the quickstart HR corpus is ~0.6 rather than the 0.9 that suits
question dedup. The absolute number that separates "same claim" from "both are Ukrainian prose"
depends on the encoder, the language, the chunk size, and how narrow the corpus's subject matter
is -- none of which the operator should have to discover by sweeping.

So calibrate against the corpus's own **null distribution**: sample random comparable chunk pairs
and read the threshold off their similarity tail. The absolute cosine that resolves to is recorded
per run, so the operator still sees the number.

A raw quantile is the wrong dial to expose, though, and the quickstart goods corpus shows why. A
quantile is a per-PAIR error rate, so the number of chance flags it admits grows with the pair
space: at the 99.9th percentile over that corpus's 74,586 comparable cross-document pairs, chance
alone predicts 75 flags -- and the calibrated run reported 84, essentially all noise. The same
quantile on a 100k-chunk corpus would admit millions. What an operator actually wants to bound is
the number of rows they have to read, which is why `max_candidate_pairs` is the headline
knob: it converts a row BUDGET into the quantile that delivers it, given this corpus's
actual pair count. `cos_quantile` stays available as the low-level escape hatch.

What "null" means here, precisely: a random cross-document pair from the comparable set. That is
overwhelmingly unrelated content, but it is not *guaranteed* unrelated -- a corpus that really is
duplicated will contribute a few genuine duplicates to the sample. This biases the estimated
quantile slightly UPWARD (a more conservative threshold), which is the safe direction: the audit
under-reports rather than flooding the operator with rows. The bias only becomes material when
duplicates are a sizeable fraction of all pairs, which the hash and lexical tiers catch first.
"""

from dataclasses import dataclass
from typing import Any

from llb.core.contracts.common import JsonObject

DEFAULT_NULL_SAMPLE_PAIRS = 200_000
"""Sampled pairs backing the estimate.

The tail quantile is what the threshold reads off, so the sample has to be large enough that the
tail is populated: at the 0.999 quantile only one sample in a thousand lands above the cutoff, so
200k pairs leave ~200 observations deciding it. Smaller samples make the threshold jump between
runs with different seeds.
"""

DEFAULT_NULL_SEED = 0
"""Fixed seed: two runs over the same corpus must resolve the same threshold."""

SUGGESTED_MAX_CANDIDATE_PAIRS = 50
"""A starting row budget, not a default -- calibration stays opt-in.

The semantic tier feeds the claim tier, which spends one model call per candidate pair, so the
budget an operator wants is whatever adjudication they can afford. 50 is a few minutes of local
claim-tier work on the quickstart corpora.
"""

MIN_NULL_PAIRS = 100
"""Below this the quantile is not an estimate of anything; calibration refuses instead of lying."""

REPORTED_QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)
"""Recorded alongside the resolved one so the operator can see the whole tail shape."""

MAX_EXHAUSTIVE_PAIRS = 5_000_000
"""Enumerate every comparable pair below this, instead of sampling.

Sampling puts a floor under the estimable tail: with N sampled pairs the finest resolvable
per-pair rate is 1/N, so a budget needing a rarer tail silently returns the sample maximum. The
quickstart HR corpus hits exactly that -- 2.4M comparable pairs against a 200k sample, where only
one sampled pair sits above cosine 0.6 and the estimated rate stops moving. The pair scan already
computes every similarity in well under a second at this size, so enumerating removes the whole
failure mode rather than papering over it.
"""


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile over an ascending list (numpy's default `linear` method).

    Implemented here rather than delegated to numpy so the resolved threshold is bit-identical
    whether or not the optional `[rag]` extra is installed -- a calibrated default that shifted
    with the environment would be worse than the fixed one it replaces.
    """
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sample")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


@dataclass(frozen=True)
class NullDistribution:
    """The sampled similarity distribution of unrelated pairs, and what it resolves a knob to."""

    similarities: list[float]
    n_pairs: int
    total_pairs: int
    seed: int
    exhaustive: bool

    def quantile(self, q: float) -> float:
        return _quantile(self.similarities, q)

    def resolvable_quantile(self) -> float:
        """The finest per-pair rate this estimate can express (1/n for a sample; exact if whole).

        Asking for a rarer tail than the sample contains does not fail loudly on its own -- the
        quantile just pins to the sample maximum -- so callers check this before trusting a
        resolved threshold.
        """
        return 1.0 if self.exhaustive else 1.0 - (1.0 / self.n_pairs)

    def quantile_for_top_n(self, budget: int) -> float:
        """The per-pair quantile whose cosine selects the `budget` most-similar pairs.

        Over an exhaustive distribution this is exact: a tail probability of `budget/total_pairs`
        cuts at the budget-th largest similarity, so the scan returns precisely that many pairs.
        That exactness is also the limitation -- it is a rank cutoff into the corpus's own
        similarity ordering, carrying no claim about how many of those pairs are real.

        It is what makes the knob portable: the same budget resolves to a stricter cosine on a
        bigger corpus, exactly where a fixed quantile would flood.
        """
        if self.total_pairs <= 0:
            return 1.0
        return max(0.0, 1.0 - (budget / self.total_pairs))

    def payload(
        self,
        resolved_quantile: float | None = None,
        max_candidate_pairs: int | None = None,
    ) -> JsonObject:
        """The record written into `summary.json` under the semantic tier."""
        out: dict[str, Any] = {
            "n_pairs": self.n_pairs,
            "total_pairs": self.total_pairs,
            "seed": self.seed,
            "exhaustive": self.exhaustive,
            "resolvable_quantile": round(self.resolvable_quantile(), 9),
            "mean": round(sum(self.similarities) / len(self.similarities), 6),
            "min": round(self.similarities[0], 6),
            "max": round(self.similarities[-1], 6),
            "quantiles": {f"{q:g}": round(self.quantile(q), 6) for q in REPORTED_QUANTILES},
        }
        if max_candidate_pairs is not None:
            out["max_candidate_pairs"] = max_candidate_pairs
        if resolved_quantile is not None:
            out["resolved_quantile"] = resolved_quantile
            out["resolved_cos_threshold"] = round(self.quantile(resolved_quantile), 6)
            # Rank position the cutoff lands at, NOT an expected-false-positive count: the
            # distribution is not an independent null (see the module docstring).
            out["selected_rank"] = round((1.0 - resolved_quantile) * self.total_pairs, 3)
        return out
