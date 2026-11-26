"""Full-book wall time from per-class page counts and measured per-call seconds.

The measured rate is the mean duration of the recognition stages. A whole-page
engine runs on the first of those stages; the later stages copy that reading, so
their duration is near zero. Multiplying the mean by the stage count recovers the
recognition wall time. The product is a selection-set projection: the pilot's
per-page time applied to the book's class counts, not a measurement of the book.
"""

from collections.abc import Mapping, Sequence


def projected_seconds(
    counts: Mapping[str, int],
    winners: Mapping[str, str],
    routings: Mapping[str, tuple[str, Mapping[str, str]]],
    seconds: Mapping[str, float],
) -> tuple[float | None, list[str]]:
    """Seconds for the configured calls, or null when a class has no rate or no count.

    ``routings`` maps a stage to ``(default engine, class overrides)``. ``winners``
    is accepted so a caller can see which engine the class selected; the routings
    are what the configuration will actually run.
    """
    del winners
    total = 0.0
    missing: list[str] = []
    for name, count in counts.items():
        class_seconds = _class_seconds(name, routings, seconds)
        if class_seconds is None:
            missing.append(name)
            continue
        total += count * class_seconds
    if missing:
        return None, missing
    return total, []


def _class_seconds(
    name: str,
    routings: Mapping[str, tuple[str, Mapping[str, str]]],
    seconds: Mapping[str, float],
) -> float | None:
    calls: dict[str, int] = {}
    for default, overrides in routings.values():
        engine = overrides.get(name, default)
        calls[engine] = calls.get(engine, 0) + 1
    total = 0.0
    for engine, count in calls.items():
        rate = seconds.get(engine)
        if rate is None:
            return None
        total += rate * count
    return total


def counts_for(classes: Sequence[str], supplied: Mapping[str, int]) -> dict[str, int] | None:
    """The supplied counts when every class is present, else None."""
    if any(name not in supplied for name in classes):
        return None
    return {name: supplied[name] for name in classes}
