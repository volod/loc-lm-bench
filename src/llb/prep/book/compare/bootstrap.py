"""The page-bootstrap interval for a class of two or three pilot pages.

The statistic is the mean of the page values. Every resample of the pages with
replacement has a mean between the worst page and the best page, and both ends are
reached by drawing that one page every time. The interval is that range. A gain is
an interval that lies entirely above zero, so every pilot page in the class moved
the same way.
"""

from collections.abc import Sequence
from typing import Literal

MarginStatus = Literal["gain", "loss", "inconclusive"]


def page_bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    """Minimum and maximum of the resampled mean. Empty input is refused."""
    if not values:
        raise ValueError("a page bootstrap needs at least one page")
    return float(min(values)), float(max(values))


def interval_status(low: float, high: float) -> MarginStatus:
    """``gain`` only when every resampled mean is positive."""
    if low > 0:
        return "gain"
    if high < 0:
        return "loss"
    return "inconclusive"
