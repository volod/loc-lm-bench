"""Fixtures shared by the benchmark unit tests."""

from collections.abc import Callable, Iterable, Iterator
from itertools import repeat

import pytest

Clock = Callable[[], float]
ClockFactory = Callable[[Iterable[float] | None], Clock]


def _episode_durations_clock(durations_s: Iterable[float]) -> Clock:
    """Return a monotonic clock whose successive episodes take the declared durations."""
    durations: Iterator[float] = iter(durations_s)
    now = 0.0
    at_start = True

    def clock() -> float:
        nonlocal at_start, now
        if at_start:
            at_start = False
            return now
        duration = float(next(durations))
        if duration < 0.0:
            raise ValueError("episode duration must be non-negative")
        now += duration
        at_start = True
        return now

    return clock


@pytest.fixture
def episode_clock() -> ClockFactory:
    """Build an exact monotonic episode clock, defaulting every episode to one second."""

    def factory(durations_s: Iterable[float] | None = None) -> Clock:
        return _episode_durations_clock(durations_s if durations_s is not None else repeat(1.0))

    return factory
