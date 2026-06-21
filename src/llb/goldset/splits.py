"""Deterministic, disjoint split assignment for gold items.

Splits (calibration / tuning / final) must be disjoint so config tuning never leaks into
the final leaderboard number. Assignment is a seeded shuffle over sorted ids, so it is
reproducible across runs and machines.
"""

import random
from collections.abc import Iterable

DEFAULT_RATIOS = {"calibration": 0.34, "tuning": 0.33, "final": 0.33}


def assign_splits(
    ids: Iterable[str],
    ratios: dict[str, float] | None = None,
    seed: int = 13,
) -> dict[str, str]:
    """Map each id to exactly one split. Deterministic for a given (ids, ratios, seed)."""
    ratios = ratios or DEFAULT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    unique = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(unique)

    names = list(ratios)
    total = len(unique)
    counts = [int(total * ratios[name]) for name in names]

    out: dict[str, str] = {}
    idx = 0
    for name, count in zip(names, counts):
        for _ in range(count):
            out[unique[idx]] = name
            idx += 1
    # distribute the truncation remainder round-robin (deterministic)
    j = 0
    while idx < total:
        out[unique[idx]] = names[j % len(names)]
        idx += 1
        j += 1
    return out
