"""Cohen and Fleiss inter-reviewer agreement metrics."""

from collections.abc import Sequence


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} != {len(b)}")
    count = len(a)
    if count == 0:
        return 0.0
    observed = sum(1 for left, right in zip(a, b) if left == right) / count
    labels = set(a) | set(b)
    expected = sum(
        (list(a).count(label) / count) * (list(b).count(label) / count) for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(counts: Sequence[Sequence[int]]) -> float:
    rows = [list(row) for row in counts if sum(row) > 0]
    if not rows:
        return 0.0
    raters = sum(rows[0])
    if raters < 2:
        raise ValueError("Fleiss' kappa needs at least 2 raters per item")
    if any(sum(row) != raters for row in rows):
        raise ValueError("every item must be rated by the same number of raters")
    item_agreement = [
        (sum(count * count for count in row) - raters) / (raters * (raters - 1)) for row in rows
    ]
    observed = sum(item_agreement) / len(rows)
    totals = [sum(row[index] for row in rows) for index in range(len(rows[0]))]
    category_rates = [total / (len(rows) * raters) for total in totals]
    expected = sum(rate * rate for rate in category_rates)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
