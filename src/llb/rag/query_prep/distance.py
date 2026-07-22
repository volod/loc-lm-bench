"""Bounded Damerau-Levenshtein (OSA) distance shared by the typo and restoration steps.

Kept separate from both so the candidate scanner (`typos`) and the surface-compatibility check
(`restore`) can use it without importing each other.
"""


def damerau_levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    """Optimal string alignment (Damerau-Levenshtein with adjacent transpositions), bounded.

    Returns `max_distance + 1` early once the running minimum exceeds `max_distance`, so the
    per-token vocabulary scan stays cheap. Exact distance when `max_distance` is None.
    """
    la, lb = len(a), len(b)
    if max_distance is not None and abs(la - lb) > max_distance:
        return max_distance + 1
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, prev2[j - 2] + 1)
            cur[j] = best
            row_best = min(row_best, best)
        if max_distance is not None and row_best > max_distance:
            return max_distance + 1
        prev2, prev = prev, cur
    return prev[lb]
