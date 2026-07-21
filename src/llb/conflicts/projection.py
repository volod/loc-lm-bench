"""Exact PCA lower-bound projection for large-corpus conflict blocking.

For unit source vectors, a cosine cutoff `c` is the Euclidean cutoff `sqrt(2 - 2c)`. An
orthogonal projection can only shrink pairwise Euclidean distance, so a projected distance above
that cutoff proves the full-space pair cannot match. Pairs that survive are confirmed against the
original vectors; the final result is therefore exact, not approximate ANN recall.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.conflicts.vectorops import METRIC_EUCLIDEAN, VectorSet

PROJECTION_VERSION = "conflict-pca-v1"


def euclidean_threshold(cos_threshold: float) -> float:
    """Euclidean distance equivalent of a cosine cutoff for unit vectors."""
    return math.sqrt(max(0.0, 2.0 - 2.0 * cos_threshold))


@dataclass(frozen=True)
class PCAProjection:
    """A fitted orthogonal projection plus the provenance that controls reuse."""

    mean: list[float]
    components: list[list[float]]
    embedding_model: str
    centered: bool
    fitted_source_fingerprint: str

    @property
    def source_dim(self) -> int:
        return len(self.mean)

    @property
    def dims(self) -> int:
        return len(self.components)

    def transform(self, vectors: VectorSet) -> VectorSet:
        """Project without normalization, preserving the contraction guarantee."""
        if vectors.dim != self.source_dim:
            raise ValueError(f"projection expects {self.source_dim} dimensions, got {vectors.dim}")
        numpy = _numpy()
        matrix = vectors.numpy_matrix()
        projected = (matrix - numpy.asarray(self.mean)) @ numpy.asarray(self.components).T
        return VectorSet.from_any(projected, metric=METRIC_EUCLIDEAN)

    def compatible(
        self, *, embedding_model: str, source_dim: int, dims: int, centered: bool
    ) -> bool:
        """Whether this orthogonal basis is safe to apply to the current vector space."""
        return (
            self.embedding_model == embedding_model
            and self.source_dim == source_dim
            and self.dims == dims
            and self.centered == centered
        )

    def payload(self) -> dict[str, Any]:
        core = {
            "version": PROJECTION_VERSION,
            "mean": self.mean,
            "components": self.components,
            "embedding_model": self.embedding_model,
            "centered": self.centered,
            "fitted_source_fingerprint": self.fitted_source_fingerprint,
        }
        return {**core, "fingerprint": _fingerprint(core)}

    @property
    def fingerprint(self) -> str:
        return str(self.payload()["fingerprint"])

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.payload(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "PCAProjection":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != PROJECTION_VERSION:
            raise ValueError("unsupported PCA projection version")
        core = {key: value for key, value in payload.items() if key != "fingerprint"}
        if payload.get("fingerprint") != _fingerprint(core):
            raise ValueError("PCA projection fingerprint mismatch")
        return cls(
            mean=[float(value) for value in payload["mean"]],
            components=[
                [float(value) for value in component] for component in payload["components"]
            ],
            embedding_model=str(payload["embedding_model"]),
            centered=bool(payload["centered"]),
            fitted_source_fingerprint=str(payload["fitted_source_fingerprint"]),
        )


def fit_pca_projection(
    vectors: VectorSet,
    dims: int,
    *,
    embedding_model: str,
    centered: bool,
    source_fingerprint: str,
) -> PCAProjection:
    """Fit deterministic top-variance orthonormal axes from a covariance eigendecomposition."""
    if not 1 <= dims <= vectors.dim:
        raise ValueError(f"projected dimensions must be in [1, {vectors.dim}], got {dims}")
    numpy = _numpy()
    matrix = vectors.numpy_matrix()
    mean = matrix.mean(axis=0) if len(vectors) else numpy.zeros(vectors.dim)
    shifted = matrix - mean
    covariance = shifted.T @ shifted
    _, eigenvectors = numpy.linalg.eigh(covariance)
    components = eigenvectors[:, -dims:].T[::-1].copy()
    _canonicalize_signs(components)
    return PCAProjection(
        mean=[float(value) for value in mean],
        components=[[float(value) for value in row] for row in components],
        embedding_model=embedding_model,
        centered=centered,
        fitted_source_fingerprint=source_fingerprint,
    )


def _canonicalize_signs(components: Any) -> None:
    """Remove eigensolver sign ambiguity so repeated fits serialize identically."""
    numpy = _numpy()
    for row in components:
        pivot = int(numpy.argmax(numpy.abs(row)))
        if row[pivot] < 0:
            row *= -1


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - numpy is in dev and rag extras
        raise RuntimeError("PCA conflict blocking requires numpy") from exc
    return numpy
