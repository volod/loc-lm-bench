"""Spread of one number over repeated measurements of the SAME configuration (pure).

A reported metric has two kinds of uncertainty and they are not the same thing. The paired
bootstrap in `stats.py` prices the ITEM SAMPLE -- "would another draw of questions have said this?"
-- and every interval in this repo is that one. It cannot see the other kind: re-running the
identical configuration on the identical items and getting a different number. Numeric drift in an
encoder's kernels moves a retrieval metric that way (`llb.rag.noise_floor.measure`), and decode
nondeterminism moves an answer metric that way
(`llb.eval.context_ablation.decoding_stability`).

Both lanes need the identical statistic over their replicate values -- the observed band, its
half-width, and the single value the artifact actually quotes -- so it lives here once rather than
being re-derived per lane. `half_width` is the "+/-" to read beside a number: a delta smaller than
it was not measured, it was observed.
"""

import statistics
from collections.abc import Sequence

from typing_extensions import TypedDict


class ValueSpread(TypedDict):
    """Band of one number across replicate measurements, plus the value the report quotes.

    `base` is the quoted value, NOT a summary of the replicates: a comparison states one run's
    numbers and this says how far a re-measurement moves them, so collapsing the two would hide
    exactly the question. It is the unjittered metric for a noise floor and the first repeat for a
    decode-stability report.
    """

    base: float
    min: float
    max: float
    mean: float
    std: float
    half_width: float  # (max - min) / 2 -- the "+/-" to read beside the metric


def value_spread(base: float, values: Sequence[float]) -> ValueSpread:
    """The band `values` occupy, stated beside the `base` value the artifact quotes."""
    if not values:
        raise ValueError("a spread needs at least one replicate value")
    low, high = min(values), max(values)
    return {
        "base": base,
        "min": low,
        "max": high,
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "half_width": (high - low) / 2.0,
    }


def format_band(spread: ValueSpread, digits: int = 3) -> str:
    """`min-max (+/-half_width)` -- the shared band cell of every floor table."""
    return (
        f"{spread['min']:.{digits}f}-{spread['max']:.{digits}f} "
        f"(+/-{spread['half_width']:.{digits}f})"
    )


__all__ = ["ValueSpread", "format_band", "value_spread"]
