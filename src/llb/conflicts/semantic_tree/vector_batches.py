"""Numpy batch operations for conflict-vector scans."""

from typing import Any


def cross_group_similarities(
    matrix: Any,
    numpy: Any,
    indices: list[int],
    groups: list[int],
    block: int,
) -> list[float]:
    """Score each unordered cross-group pair without materializing the pair list."""
    selected = matrix[indices]
    labels = numpy.asarray(groups)
    row_index = numpy.arange(len(indices))
    out: list[float] = []
    for start in range(0, len(indices), block):
        stop = min(start + block, len(indices))
        similarities = selected[start:stop] @ selected.T
        rows = row_index[start:stop][:, None]
        keep = (rows < row_index[None, :]) & (labels[start:stop][:, None] != labels[None, :])
        out.extend(float(value) for value in similarities[keep])
    return out
