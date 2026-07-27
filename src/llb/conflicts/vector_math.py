"""Scalar vector geometry shared by conflict indexes and trees."""

import math

Vector = list[float]
METRIC_ANGULAR = "angular"
METRIC_EUCLIDEAN = "euclidean"
METRICS = (METRIC_ANGULAR, METRIC_EUCLIDEAN)


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def angular_distance(cosine: float) -> float:
    """Angular distance in radians for a cosine similarity."""
    return math.acos(max(-1.0, min(1.0, cosine)))


def vector_distance(metric: str, a: Vector, b: Vector) -> float:
    """Distance between arbitrary vectors under a supported tree metric."""
    if metric == METRIC_ANGULAR:
        return angular_distance(dot(a, b))
    if metric == METRIC_EUCLIDEAN:
        return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))
    raise ValueError(f"unknown vector metric {metric!r}")
