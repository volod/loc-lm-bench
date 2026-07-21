"""Vector primitives for the semantic prefix tree, with optional numpy acceleration.

The tree algorithms above this module are backend-agnostic: they only ask a `VectorSet` for dot
products and centroids. numpy is used when importable (every CUDA host and the full local install
have it via the `[rag]` extra) and a pure-Python path covers the base `[dev]` environment GitHub
CI runs, so the tree and its tests never need an optional extra. Both paths compute the same
values; only the speed differs.

Vectors are L2-normalized on load, so a dot product IS the cosine similarity -- the same identity
FAISS relies on for its inner-product index.
"""

import math
from typing import Any

Vector = list[float]
METRIC_ANGULAR = "angular"
METRIC_EUCLIDEAN = "euclidean"
METRICS = (METRIC_ANGULAR, METRIC_EUCLIDEAN)


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError:
        return None
    return numpy


class VectorSet:
    """An immutable row-vector set addressed by build-order ordinal.

    Encoder vectors use angular geometry and are L2-normalized. PCA projections use Euclidean
    geometry and MUST retain their original lengths: re-normalizing a projection destroys the
    lower-bound guarantee that makes projected conflict blocking exact.
    """

    def __init__(
        self,
        rows: list[Vector],
        use_numpy: bool = True,
        *,
        metric: str = METRIC_ANGULAR,
    ):
        if metric not in METRICS:
            raise ValueError(f"unknown vector metric {metric!r}")
        self._np = _import_numpy() if use_numpy else None
        self.metric = metric
        prepared = (
            [_normalize(row) for row in rows]
            if metric == METRIC_ANGULAR
            else [list(row) for row in rows]
        )
        self.dim = len(prepared[0]) if prepared else 0
        for row in prepared:
            if len(row) != self.dim:
                raise ValueError("every vector must have the same dimension")
        self._rows = prepared
        self._matrix = self._np.asarray(prepared, dtype="float64") if self._np else None

    @classmethod
    def from_any(
        cls,
        vectors: Any,
        use_numpy: bool = True,
        *,
        metric: str = METRIC_ANGULAR,
    ) -> "VectorSet":
        """Build from a numpy matrix, a list of lists, or anything with `tolist()`."""
        rows = vectors.tolist() if hasattr(vectors, "tolist") else [list(row) for row in vectors]
        return cls(
            [[float(value) for value in row] for row in rows],
            use_numpy=use_numpy,
            metric=metric,
        )

    def __len__(self) -> int:
        return len(self._rows)

    def row(self, index: int) -> Vector:
        return self._rows[index]

    def similarity(self, left: int, right: int) -> float:
        """Cosine similarity between two stored rows."""
        return _dot(self._rows[left], self._rows[right])

    def similarity_to(self, vector: Vector, indices: list[int]) -> list[float]:
        """Cosine similarity of `vector` against each row in `indices`, in order."""
        if self._np is not None and self._matrix is not None and indices:
            block = self._matrix[indices]
            return [float(value) for value in block @ self._np.asarray(vector, dtype="float64")]
        return [_dot(vector, self._rows[index]) for index in indices]

    def distance(self, left: int, right: int) -> float:
        """Metric distance between two rows."""
        return vector_distance(self.metric, self._rows[left], self._rows[right])

    def distances_to(self, vector: Vector, indices: list[int]) -> list[float]:
        """Metric distance from `vector` to each selected row."""
        if self._np is not None and self._matrix is not None and indices:
            block = self._matrix[indices]
            target = self._np.asarray(vector, dtype="float64")
            if self.metric == METRIC_EUCLIDEAN:
                return [float(value) for value in self._np.linalg.norm(block - target, axis=1)]
            dots = block @ target
            return [angular_distance(float(value)) for value in dots]
        return [vector_distance(self.metric, vector, self._rows[index]) for index in indices]

    def pair_similarities(self, pairs: list[tuple[int, int]]) -> list[float]:
        """Cosine similarity for an arbitrary list of ordinal pairs, in the order given.

        Unlike `pairs_above` this scores a chosen SUBSET rather than scanning every pair, which
        is what null-distribution sampling needs: a few hundred thousand random pairs out of a
        pair space too large to materialize. The numpy path gathers both sides and takes a
        row-wise dot product, so cost is linear in the sample rather than quadratic in the corpus.
        """
        if not pairs:
            return []
        if self._np is not None and self._matrix is not None:
            left = self._matrix[[pair[0] for pair in pairs]]
            right = self._matrix[[pair[1] for pair in pairs]]
            return [float(value) for value in (left * right).sum(axis=1)]
        return [_dot(self._rows[left], self._rows[right]) for left, right in pairs]

    def pairs_above_candidates(
        self,
        pairs: list[tuple[int, int]],
        threshold: float,
        *,
        block: int = 16_384,
    ) -> list[tuple[int, int, float]]:
        """Exactly confirm chosen pairs in bounded batches.

        Gathering both sides of every projected candidate at once can consume several gigabytes
        on a large corpus. Batching bounds temporary memory while retaining numpy's vectorized
        full-space confirmation.
        """
        out: list[tuple[int, int, float]] = []
        for start in range(0, len(pairs), block):
            selected = pairs[start : start + block]
            similarities = self.pair_similarities(selected)
            out.extend(
                (left, right, similarity)
                for (left, right), similarity in zip(selected, similarities)
                if similarity >= threshold
            )
        return out

    def cross_group_similarities(self, indices: list[int], groups: list[int]) -> list[float]:
        """Similarity of every pair of `indices` whose `groups` label differs, unordered.

        `groups[i]` labels `indices[i]` (the conflict audit passes document ids), so this is the
        exact set of pairs the cross-document scan considers -- which is what makes it usable as
        an EXHAUSTIVE null distribution rather than a sample of one. The numpy path never
        materializes the pair list, only one block of the similarity matrix at a time.
        """
        if len(indices) != len(groups):
            raise ValueError("indices and groups must be the same length")
        if self._np is not None and self._matrix is not None:
            return self._cross_group_numpy(indices, groups)
        return [
            _dot(self._rows[indices[i]], self._rows[indices[j]])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
            if groups[i] != groups[j]
        ]

    def _cross_group_numpy(
        self, indices: list[int], groups: list[int], block: int = 512
    ) -> list[float]:
        numpy = self._np
        if numpy is None or self._matrix is None:  # pragma: no cover - guarded by the caller
            raise RuntimeError("numpy path requested without a numpy matrix")
        matrix = self._matrix[indices]
        labels = numpy.asarray(groups)
        row_index = numpy.arange(len(indices))
        out: list[float] = []
        for start in range(0, len(indices), block):
            stop = min(start + block, len(indices))
            similarities = matrix[start:stop] @ matrix.T
            # Upper triangle only (each unordered pair once) and different groups only.
            rows = row_index[start:stop][:, None]
            keep = (rows < row_index[None, :]) & (labels[start:stop][:, None] != labels[None, :])
            out.extend(float(value) for value in similarities[keep])
        return out

    def centered(self) -> "VectorSet":
        """This set with the corpus mean direction removed, then renormalized.

        Sentence-encoder spaces are strongly anisotropic: on a real Ukrainian corpus every
        multilingual-E5 chunk vector sits in a narrow cone, so two COMPLETELY unrelated chunks
        still score cosine 0.83 and a 0.9 "near-duplicate" threshold is barely above noise.
        Subtracting the mean (the standard all-but-the-top correction) restores an isotropic
        space where unrelated pairs score about 0 and a threshold means what it says.
        """
        if not self._rows:
            return VectorSet([], use_numpy=self._np is not None, metric=self.metric)
        mean = [sum(column) / len(self._rows) for column in zip(*self._rows)]
        shifted = [[value - offset for value, offset in zip(row, mean)] for row in self._rows]
        return VectorSet(shifted, use_numpy=self._np is not None, metric=self.metric)

    def pairs_above(self, threshold: float, *, block: int = 512) -> list[tuple[int, int, float]]:
        """Every pair at or above `threshold`, as sorted `(low, high, similarity)`.

        This remains the small-corpus path and the exact baseline for projected blocking. The
        large-corpus path uses a PCA Euclidean lower bound, then confirms its candidates against
        these full-space vectors.
        """
        if self.metric != METRIC_ANGULAR:
            raise ValueError("pairs_above requires angular vectors")
        if self._matrix is not None:
            return self._pairs_above_numpy(threshold, self._matrix, block)
        return [
            (left, right, similarity)
            for left in range(len(self._rows))
            for right in range(left + 1, len(self._rows))
            if (similarity := _dot(self._rows[left], self._rows[right])) >= threshold
        ]

    def _pairs_above_numpy(
        self, threshold: float, matrix: Any, block: int
    ) -> list[tuple[int, int, float]]:
        out: list[tuple[int, int, float]] = []
        total = len(self._rows)
        for start in range(0, total, block):
            similarities = matrix[start : start + block] @ matrix.T
            rows, columns = (similarities >= threshold).nonzero()
            for row, column in zip(rows.tolist(), columns.tolist()):
                left = start + row
                if left < column:
                    out.append((left, column, float(similarities[row][column])))
        return sorted(out)

    def centroid(self, indices: list[int]) -> Vector:
        """The normalized mean of the given rows (a zero mean falls back to the first row)."""
        if not indices:
            raise ValueError("centroid needs at least one member")
        if self._matrix is not None:
            mean = self._matrix[indices].mean(axis=0)
            values = [float(value) for value in mean]
            return _normalize(values) if self.metric == METRIC_ANGULAR else values
        total = [0.0] * self.dim
        for index in indices:
            row = self._rows[index]
            for position in range(self.dim):
                total[position] += row[position]
        mean = [value / len(indices) for value in total]
        if self.metric == METRIC_EUCLIDEAN:
            return mean
        normalized = _normalize(mean)
        return normalized if any(normalized) else list(self._rows[indices[0]])

    def numpy_matrix(self) -> Any:
        """The internal read-only numerical matrix, or a freshly imported numpy copy."""
        numpy = self._np or _import_numpy()
        if numpy is None:
            raise RuntimeError("PCA conflict blocking requires numpy")
        return (
            self._matrix if self._matrix is not None else numpy.asarray(self._rows, dtype="float64")
        )


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def angular_distance(cosine: float) -> float:
    """Angular distance in radians for a cosine similarity.

    Unlike cosine similarity, angular distance is a true metric, so the triangle inequality holds
    -- which is what makes the tree's pruning bound sound rather than heuristic.
    """
    return math.acos(max(-1.0, min(1.0, cosine)))


def vector_distance(metric: str, a: Vector, b: Vector) -> float:
    """Distance between arbitrary vectors under a supported tree metric."""
    if metric == METRIC_ANGULAR:
        return angular_distance(_dot(a, b))
    if metric == METRIC_EUCLIDEAN:
        return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))
    raise ValueError(f"unknown vector metric {metric!r}")
