"""Choosing ONE prompt guard out of a predeclared band, shared by every per-family guard fit.

A context-policy study that holds its prompt guard fixed across families is holding a CHARACTER
constant fixed, not the regime it wanted: the same guard folds at a different step for a family
whose summaries run long, and never folds at all for one whose walk ends early. The fix each such
study reaches for is the same -- declare a band of candidate guards up front, score every guard in
it against what that family actually measured, and take the best one -- so the band arithmetic
lives here and only the SCORE differs per study.

The two rules the selection encodes are what make a fit auditable rather than convenient:

  - the DECLARED guard wins any tie, so a family the shared constant already suits keeps the
    published geometry and only the family it does not suit moves -- a fit that shuffled every
    family's guard would invalidate the comparison it exists to enable;
  - among the rest, the middle of the widest contiguous run of best-scoring guards wins, which is
    the guard furthest from the edge where the score changes, so a family whose next run measures
    slightly differently still lands on the same choice.

Nothing here decides what a guard is scored ON. A fold-count ladder scores it by how many measured
fold lengths land on its target rung; an elision study scores it by how many cases fold within the
walk their family actually takes. Both hand this module a `{guard: score}` mapping.
"""

from llb.bench.agentic.design_fields import as_int

# The keys a design's fit block states its candidate band with, inclusive and ascending.
BAND_MIN_FIELD = "search_min_chars"
BAND_MAX_FIELD = "search_max_chars"
BAND_STEP_FIELD = "step_chars"


def search_band(spec: dict[str, object]) -> tuple[int, int, int]:
    """The predeclared candidate guards, as an inclusive ascending range with a step."""
    return (
        as_int(spec, BAND_MIN_FIELD),
        as_int(spec, BAND_MAX_FIELD),
        as_int(spec, BAND_STEP_FIELD),
    )


def guard_grid(spec: dict[str, object]) -> list[int]:
    """Every candidate guard the declared band contains."""
    low, high, step = search_band(spec)
    if step < 1 or high < low:
        raise ValueError(f"a guard band needs an ascending range and a positive step, got {spec!r}")
    return list(range(low, high + 1, step))


def select_guard(scores: dict[int, int], declared: int) -> int:
    """The best-scoring guard: the declared one on a tie, else the centre of the widest run."""
    if not scores:
        raise ValueError("a guard fit needs at least one candidate guard to choose from")
    best = max(scores.values())
    if scores.get(declared) == best:
        return declared
    return _widest_run_centre(scores, best)


def median_int(values: list[int]) -> int:
    """The upper median of a measured list, or 0 when nothing was measured."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _widest_run_centre(scores: dict[int, int], best: int) -> int:
    """The middle of the longest contiguous stretch of best-scoring guards.

    Contiguity is measured over the CANDIDATES rather than in characters: two guards are
    neighbours when the mapping lists them consecutively, so a band that skipped a guard as
    unusable breaks the run there, which is exactly what it should do.
    """
    guards = sorted(scores)
    runs: list[list[int]] = []
    for index, guard in enumerate(guards):
        if scores[guard] != best:
            continue
        if index > 0 and scores.get(guards[index - 1]) == best and runs:
            runs[-1].append(guard)
        else:
            runs.append([guard])
    widest = max(runs, key=len)
    return widest[len(widest) // 2]
