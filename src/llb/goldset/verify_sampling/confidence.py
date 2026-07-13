"""Heuristic confidence ordering for the human review queue."""

from collections.abc import Sequence


def row_confidence(row: dict[str, str]) -> float:
    """Compute a prior plausibility score from read-only worksheet signals."""
    score = 0.0
    for column in ("cc_grounded", "cc_non_circular", "cc_supported", "cc_answerable"):
        value = (row.get(column) or "").strip().lower()
        if value == "true":
            score += 1.0
        elif value == "false":
            score -= 1.0
    rank = (row.get("retrieval_rank") or "").strip()
    if rank.isdigit() and int(rank) > 0:
        score += 1.0 / int(rank)
    return score


def confidence_order(rows: Sequence[dict[str, str]]) -> list[int]:
    """Return row indices from least confident to most confident."""
    return sorted(range(len(rows)), key=lambda index: (row_confidence(rows[index]), index))
