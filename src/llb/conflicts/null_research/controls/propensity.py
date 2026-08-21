"""Cross-fitted propensity balancing between a target corpus and its control banks.

Nearest-neighbour matching (second generation) reuses a few reference chunks many times and still
leaves corpus membership predictable. This module replaces it with the standard covariate-shift
correction: fit `P(target | covariates)` out of fold, then weight EVERY reference chunk once by its
odds so the weighted reference covariate distribution reproduces the target's. Each reference chunk
therefore contributes at most one weighted observation, which is what makes an effective-sample-size
statement about the control bank meaningful.

The covariates deliberately span both requested families: the structural surface features the
matched lane already used, plus encoder-neighbourhood features (affinity to the target cloud and
the leading principal directions of the target space) that a purely structural match cannot see.
"""

from dataclasses import dataclass

import numpy as np

from llb.conflicts.null_research.geometry import CorpusGeometry
from llb.conflicts.null_research.controls.matching import SURFACE_FEATURE_NAMES, surface_features
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord

BALANCE_FEATURE_NAMES = (
    *SURFACE_FEATURE_NAMES,
    "target_neighborhood_affinity",
    "principal_component_1",
    "principal_component_2",
)
NEIGHBORHOOD_K = 5
PRINCIPAL_COMPONENTS = 2
NEWTON_STEPS = 25
RIDGE = 1e-3
MAX_ODDS_RATIO = 20.0
BALANCE_FOLDS = 2


@dataclass(frozen=True)
class TargetSpace:
    """The target corpus's centered geometry plus the basis its covariates are read in."""

    matrix: np.ndarray
    mean_vector: list[float]
    components: np.ndarray

    def project(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(rows @ self.components.T)


@dataclass(frozen=True)
class PropensityWeights:
    """Out-of-fold odds weights for one reference bank, plus its balance diagnostics."""

    weights: np.ndarray
    target_scores: np.ndarray
    reference_scores: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    coefficients: list[np.ndarray]
    diagnostics: JsonObject

    @property
    def effective_sample_size(self) -> float:
        total = float(self.weights.sum())
        square = float((self.weights**2).sum())
        return total * total / square if square > 0.0 else 0.0

    def score(self, features: np.ndarray) -> np.ndarray:
        """Apply the cross-fitted membership model to rows it was never fitted on."""
        design = _design((features - self.center) / self.scale)
        return np.asarray(
            np.mean([design @ coefficients for coefficients in self.coefficients], axis=0)
        )

    def odds_weights(self, scores: np.ndarray) -> np.ndarray:
        """Trimmed, mean-one odds weights for an arbitrary scored reference population."""
        return _clip_odds(np.exp(np.clip(scores, -30.0, 30.0)))


def target_space(target: CorpusGeometry) -> TargetSpace:
    """Centered target rows and the leading directions used as encoder-neighbourhood covariates."""
    matrix = np.asarray([target.vectors.row(index) for index in target.allowed], dtype="float64")
    deviation = matrix - matrix.mean(axis=0)
    _, _, right = np.linalg.svd(deviation, full_matrices=False)
    return TargetSpace(
        matrix=matrix,
        mean_vector=list(target.raw_vectors.mean_vector()),
        components=np.ascontiguousarray(right[:PRINCIPAL_COMPONENTS]),
    )


def _neighborhood_affinity(rows: np.ndarray, space: TargetSpace, *, drop_self: bool) -> np.ndarray:
    similarities = rows @ space.matrix.T
    take = min(NEIGHBORHOOD_K + (1 if drop_self else 0), similarities.shape[1])
    top = np.sort(np.partition(similarities, -take, axis=1)[:, -take:], axis=1)[:, ::-1]
    return np.asarray(top[:, 1:].mean(axis=1) if drop_self else top.mean(axis=1))


def balance_features(
    geometry: CorpusGeometry,
    space: TargetSpace,
    *,
    rows: np.ndarray,
    is_target: bool,
) -> np.ndarray:
    """Structural covariates and encoder-neighbourhood covariates for one chunk population."""
    chunks: list[ChunkRecord] = [geometry.chunks[index] for index in geometry.allowed]
    structural = np.asarray(
        [
            surface_features(chunk, geometry.raw_vectors.row(index), space.mean_vector)
            for chunk, index in zip(chunks, geometry.allowed)
        ],
        dtype="float64",
    )
    affinity = _neighborhood_affinity(rows, space, drop_self=is_target)
    return np.column_stack((structural, affinity, space.project(rows)))


def _standardize(
    target: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.concatenate((target, reference))
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (target - center) / scale, (reference - center) / scale, center, scale


def _fit_logistic(design: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Ridge-penalised Newton (IRLS) fit; deterministic and small enough to solve exactly."""
    weights = np.zeros(design.shape[1], dtype="float64")
    penalty = RIDGE * np.eye(design.shape[1])
    for _ in range(NEWTON_STEPS):
        probability = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (labels - probability) - RIDGE * weights
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        hessian = design.T @ (design * variance[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights = weights + step
        if float(np.abs(step).max()) < 1e-9:
            break
    return weights


def _design(rows: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(rows)), rows))


def weighted_rank_auc(
    positive: np.ndarray,
    negative: np.ndarray,
    negative_weights: np.ndarray,
) -> float:
    """Rank AUC in which each negative carries its balancing weight; ties score one half."""
    total_negative = float(negative_weights.sum())
    if len(positive) == 0 or total_negative <= 0.0:
        return 1.0
    order = np.argsort(negative, kind="stable")
    sorted_negative = negative[order]
    cumulative = np.concatenate(([0.0], np.cumsum(negative_weights[order])))
    below = cumulative[np.searchsorted(sorted_negative, positive, side="left")]
    equal = (
        cumulative[np.searchsorted(sorted_negative, positive, side="right")]
        - cumulative[np.searchsorted(sorted_negative, positive, side="left")]
    )
    return float((below + 0.5 * equal).sum() / (len(positive) * total_negative))


def _weighted_standardized_differences(
    target: np.ndarray, reference: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    weighted_mean = (reference * weights[:, None]).sum(axis=0) / weights.sum()
    pooled_scale = np.concatenate((target, reference)).std(axis=0)
    pooled_scale[pooled_scale < 1e-12] = 1.0
    return np.asarray(np.abs((target.mean(axis=0) - weighted_mean) / pooled_scale))


def _clip_odds(odds: np.ndarray) -> np.ndarray:
    positive = np.clip(odds, 1e-9, None)
    ceiling = float(np.median(positive)) * MAX_ODDS_RATIO
    clipped = np.clip(positive, None, max(ceiling, 1e-9))
    return np.asarray(clipped * (len(clipped) / clipped.sum()))


def fit_propensity_weights(
    target_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    folds: int = BALANCE_FOLDS,
) -> PropensityWeights:
    """Cross-fit the membership model, then weight references by their out-of-fold odds."""
    target, reference, center, scale = _standardize(target_features, reference_features)
    target_scores = np.zeros(len(target))
    reference_scores = np.zeros(len(reference))
    odds = np.ones(len(reference))
    fitted: list[np.ndarray] = []
    for fold in range(folds):
        target_test = np.arange(len(target)) % folds == fold
        reference_test = np.arange(len(reference)) % folds == fold
        design = _design(np.concatenate((target[~target_test], reference[~reference_test])))
        labels = np.concatenate(
            (np.ones(int((~target_test).sum())), np.zeros(int((~reference_test).sum())))
        )
        coefficients = _fit_logistic(design, labels)
        fitted.append(coefficients)
        target_scores[target_test] = _design(target[target_test]) @ coefficients
        held_out = _design(reference[reference_test]) @ coefficients
        reference_scores[reference_test] = held_out
        odds[reference_test] = np.exp(np.clip(held_out, -30.0, 30.0))
    weights = _clip_odds(odds)
    balanced = _weighted_standardized_differences(target_features, reference_features, weights)
    unbalanced = _weighted_standardized_differences(
        target_features, reference_features, np.ones(len(reference_features))
    )
    return PropensityWeights(
        weights=weights,
        target_scores=target_scores,
        reference_scores=reference_scores,
        center=center,
        scale=scale,
        coefficients=fitted,
        diagnostics={
            "balancer": "cross-fitted ridge logistic propensity, odds weighting",
            "feature_names": list(BALANCE_FEATURE_NAMES),
            "folds": folds,
            "unweighted_membership_auc": round(
                max(
                    weighted_rank_auc(target_scores, reference_scores, np.ones(len(reference))),
                    1.0
                    - weighted_rank_auc(target_scores, reference_scores, np.ones(len(reference))),
                ),
                6,
            ),
            "unweighted_max_abs_standardized_difference": round(float(unbalanced.max()), 6),
            "max_abs_standardized_difference": round(float(balanced.max()), 6),
            "max_weight": round(float(weights.max()), 6),
        },
    )
