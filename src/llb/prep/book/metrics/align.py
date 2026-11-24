"""Character and word edit distance, and the missing and duplicated word counts.

Missing words are the reference words absent from the hypothesis multiset. Duplicated words are
the hypothesis words beyond that multiset, whether a repeated reference word or a word the
reference never had. Character and word error are the edit distance divided by the reference
length, and neither total includes a dangerous-substitution penalty.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TextScore:
    """Error rates for one page. Rates are None when the reference side is empty."""

    character_error_rate: float | None
    word_error_rate: float | None
    missing_words: int
    duplicated_words: int
    reference_chars: int
    reference_words: int


def text_score(reference: str, hypothesis: str) -> TextScore:
    """Character error, word error, and the two word-multiset counts."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    missing, duplicated = _multiset_delta(ref_words, hyp_words)
    return TextScore(
        character_error_rate=_rate(edit_distance(reference, hypothesis), len(reference)),
        word_error_rate=_rate(edit_distance(ref_words, hyp_words), len(ref_words)),
        missing_words=missing,
        duplicated_words=duplicated,
        reference_chars=len(reference),
        reference_words=len(ref_words),
    )


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance. A string is a sequence of characters."""
    if reference == hypothesis:
        return 0
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for index, left in enumerate(reference, 1):
        current = [index]
        for inner, right in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[inner] + 1,
                    current[inner - 1] + 1,
                    previous[inner - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def alignment(reference: str, hypothesis: str) -> list[tuple[str, int, int]]:
    """Edit script as ``(op, ref_index, hyp_index)``. An absent side uses ``-1``.

    Equal cost prefers a substitution over a deletion plus an insertion, so a one-character
    change is one substitution. Identical strings are one equal step per character.
    """
    if reference == hypothesis:
        return [("eq", index, index) for index in range(len(reference))]
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for index in range(1, rows):
        cost[index][0] = index
    for index in range(1, cols):
        cost[0][index] = index
    for left in range(1, rows):
        for right in range(1, cols):
            replace = cost[left - 1][right - 1] + (reference[left - 1] != hypothesis[right - 1])
            cost[left][right] = min(cost[left - 1][right] + 1, cost[left][right - 1] + 1, replace)
    script: list[tuple[str, int, int]] = []
    left, right = len(reference), len(hypothesis)
    while left or right:
        left, right = _consume(cost, reference, hypothesis, left, right, script)
    script.reverse()
    return script


def _consume(
    cost: list[list[int]],
    reference: str,
    hypothesis: str,
    left: int,
    right: int,
    script: list[tuple[str, int, int]],
) -> tuple[int, int]:
    """One backtrace step. Equal characters win, then a substitution, then a deletion."""
    if _equal_step(cost, reference, hypothesis, left, right):
        script.append(("eq", left - 1, right - 1))
        return left - 1, right - 1
    if _substitution_step(cost, left, right):
        script.append(("sub", left - 1, right - 1))
        return left - 1, right - 1
    if _deletion_step(cost, left, right):
        script.append(("del", left - 1, -1))
        return left - 1, right
    script.append(("ins", -1, right - 1))
    return left, right - 1


def _equal_step(
    cost: list[list[int]], reference: str, hypothesis: str, left: int, right: int
) -> bool:
    if left == 0 or right == 0:
        return False
    if cost[left][right] != cost[left - 1][right - 1]:
        return False
    return reference[left - 1] == hypothesis[right - 1]


def _substitution_step(cost: list[list[int]], left: int, right: int) -> bool:
    if left == 0 or right == 0:
        return False
    return cost[left][right] == cost[left - 1][right - 1] + 1


def _deletion_step(cost: list[list[int]], left: int, right: int) -> bool:
    if left == 0:
        return False
    return cost[left][right] == cost[left - 1][right] + 1


def _rate(distance: int, length: int) -> float | None:
    if length == 0:
        return None
    return distance / length


def _multiset_delta(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    pool: dict[str, int] = {}
    for word in hypothesis:
        pool[word] = pool.get(word, 0) + 1
    missing = 0
    for word in reference:
        if pool.get(word, 0):
            pool[word] -= 1
        else:
            missing += 1
    return missing, sum(pool.values())
