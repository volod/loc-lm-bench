"""Propensity-balanced control null with two-way clustered tail inference."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llb.conflicts.null_research_clusters import two_way_tail_interval
from llb.conflicts.null_research_evaluation import (
    MIN_COVERAGE_PROBABILITY,
    MIN_TAIL_OBSERVATIONS,
    simulated_wilson_coverage,
    wilson_interval,
)
from llb.conflicts.null_research_geometry import CorpusGeometry
from llb.conflicts.null_research_matching import MAX_ABS_STANDARDIZED_DIFFERENCE, MAX_MEMBERSHIP_AUC
from llb.conflicts.null_research_propensity import (
    PropensityWeights,
    balance_features,
    fit_propensity_weights,
    target_space,
    weighted_rank_auc,
)
from llb.core.contracts.common import JsonObject

MAX_TWO_WAY_TAIL_FACTOR = 2.0


@dataclass(frozen=True)
class BalancedControls:
    """Every reference chunk used once, weighted so the covariate shift is corrected."""

    scores: np.ndarray
    weights: np.ndarray
    reference_raw: np.ndarray
    source_texts: int
    propensity: PropensityWeights
    domain_transfer: JsonObject

    @property
    def effective_units(self) -> int:
        return int(min(self.source_texts, self.propensity.effective_sample_size))

    def cluster_maxima(self) -> list[float]:
        return sorted(float(value) for value in self.scores.max(axis=1))

    def weighted_threshold(self, fpr: float) -> float:
        flat = self.scores.reshape(-1)
        order = np.argsort(flat, kind="stable")
        weights = np.tile(self.weights, self.scores.shape[0])[order]
        cumulative = np.cumsum(weights) / weights.sum()
        return float(flat[order][int(np.searchsorted(cumulative, 1.0 - fpr, side="left"))])

    def weighted_tail_rate(self, threshold: float) -> float:
        exceedances = (self.scores >= threshold) @ self.weights
        return float(exceedances.sum() / (self.scores.shape[0] * self.weights.sum()))


def _reference_matrix(target: CorpusGeometry, reference: CorpusGeometry) -> np.ndarray:
    vectors = (
        reference.raw_vectors.centered(target.center_mean)
        if target.center_mean is not None
        else reference.raw_vectors
    )
    return np.asarray([vectors.row(index) for index in reference.allowed], dtype="float64")


def raw_rows(corpus: CorpusGeometry) -> np.ndarray:
    """Untransformed store vectors for the comparable chunks, in `allowed` order."""
    return np.asarray([corpus.raw_vectors.row(index) for index in corpus.allowed], dtype="float64")


def _domain_transfer(features: dict[str, np.ndarray], target_features: np.ndarray) -> JsonObject:
    """Leave-one-domain-out: balance on the other domains, then score the held-out domain.

    Balancing weights that only work on the domains they were fitted on are overfitting, not
    exchangeability, so the held-out domain is scored by the OTHER domains' fitted model.
    """
    held_out: dict[str, float] = {}
    for name in features:
        others = [rows for other, rows in features.items() if other != name]
        if not others:
            continue
        fitted = fit_propensity_weights(target_features, np.concatenate(others))
        transferred_scores = fitted.score(features[name])
        auc = weighted_rank_auc(
            fitted.target_scores,
            transferred_scores,
            fitted.odds_weights(transferred_scores),
        )
        held_out[name] = round(max(auc, 1.0 - auc), 6)
    return {
        "held_out_domain_membership_auc": held_out,
        "max_membership_auc": round(max(held_out.values()), 6) if held_out else 1.0,
        "domains_exchangeable": bool(held_out)
        and all(value <= MAX_MEMBERSHIP_AUC for value in held_out.values()),
    }


def build_balanced_controls(
    target: CorpusGeometry, references: dict[str, CorpusGeometry]
) -> BalancedControls:
    """Weight the whole control bank so each reference chunk contributes one balanced row."""
    if not references:
        raise ValueError("balanced controls need at least one reference corpus")
    space = target_space(target)
    target_features = balance_features(target, space, rows=space.matrix, is_target=True)
    matrices = {name: _reference_matrix(target, corpus) for name, corpus in references.items()}
    features = {
        name: balance_features(corpus, space, rows=matrices[name], is_target=False)
        for name, corpus in references.items()
    }
    names = sorted(references)
    reference_features = np.concatenate([features[name] for name in names])
    reference_matrix = np.concatenate([matrices[name] for name in names])
    propensity = fit_propensity_weights(target_features, reference_features)
    return BalancedControls(
        scores=space.matrix @ reference_matrix.T,
        weights=propensity.weights,
        reference_raw=np.concatenate([raw_rows(references[name]) for name in names]),
        source_texts=len({target.chunks[index]["text"] for index in target.allowed}),
        propensity=propensity,
        domain_transfer=_domain_transfer(features, target_features),
    )


def balanced_tail_payload(
    controls: BalancedControls, threshold: float, fpr: float, seed: int
) -> JsonObject:
    """Weighted tail statistics with pair-row and two-way clustered uncertainty side by side."""
    rate = controls.weighted_tail_rate(threshold)
    rows = controls.scores.size
    pair_lower, pair_upper = wilson_interval(int(round(rate * rows)), rows)
    cluster_maxima = controls.cluster_maxima()
    cluster_exceedances = sum(value >= threshold for value in cluster_maxima)
    effective_units = controls.effective_units
    coverage = simulated_wilson_coverage(effective_units, fpr, seed=seed)
    expected = fpr * effective_units
    bootstrap = two_way_tail_interval(
        (controls.scores >= threshold).astype("float64"), controls.weights, seed
    )
    pair_width = pair_upper - pair_lower
    two_way_width = float(bootstrap["tail_rate_two_way_width"])
    return {
        "n": rows,
        "weighted_tail_rate": round(rate, 9),
        "nominal_fpr": fpr,
        "pair_row_tail_wilson_95": [round(pair_lower, 9), round(pair_upper, 9)],
        "source_clusters": len(cluster_maxima),
        "cluster_exceedances": cluster_exceedances,
        "reference_effective_sample_size": round(controls.propensity.effective_sample_size, 3),
        "effective_independent_units": effective_units,
        "expected_independent_tail_observations": round(expected, 3),
        "wilson_coverage_simulations": round(coverage, 6),
        "coverage_valid": coverage >= MIN_COVERAGE_PROBABILITY,
        "pair_row_width_understatement": round(two_way_width / pair_width, 3)
        if pair_width > 0.0
        else None,
        **bootstrap,
        "two_way_uncertainty_resolved": float(bootstrap["tail_rate_two_way_95"][1])
        <= MAX_TWO_WAY_TAIL_FACTOR * fpr,
        "tail_resolved": expected >= MIN_TAIL_OBSERVATIONS,
    }


def score_separability(
    observed: Sequence[float] | np.ndarray,
    control_scores: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Can a threshold tell a control pair from a same-corpus pair by SCORE alone?

    Balanced covariates are a means; what a null needs is that its scores are exchangeable with the
    unrelated part of the observed population. Because genuine relations are a tiny fraction of the
    pair space, an AUC far above 0.5 here is corpus shift, not detection.
    """
    rows = control_scores.shape[0]
    return max(
        weighted_rank_auc(
            np.asarray(observed, dtype="float64"),
            control_scores.reshape(-1),
            np.tile(weights, rows),
        ),
        1.0
        - weighted_rank_auc(
            np.asarray(observed, dtype="float64"),
            control_scores.reshape(-1),
            np.tile(weights, rows),
        ),
    )


def balanced_diagnostics(
    controls: BalancedControls, observed: Sequence[float] | np.ndarray
) -> JsonObject:
    balanced = float(controls.propensity.diagnostics["max_abs_standardized_difference"])
    weighted_auc = weighted_rank_auc(
        controls.propensity.target_scores,
        controls.propensity.reference_scores,
        controls.weights,
    )
    oriented = max(weighted_auc, 1.0 - weighted_auc)
    separability = score_separability(observed, controls.scores, controls.weights)
    return {
        **controls.propensity.diagnostics,
        **controls.domain_transfer,
        "membership_auc": round(oriented, 6),
        "score_separability_auc": round(separability, 6),
        "control_rows": int(controls.scores.size),
        "unique_reference_claims": int(len(controls.weights)),
        "unique_source_texts": controls.source_texts,
        "exchangeable": bool(
            oriented <= MAX_MEMBERSHIP_AUC
            and balanced <= MAX_ABS_STANDARDIZED_DIFFERENCE
            and separability <= MAX_MEMBERSHIP_AUC
            and controls.domain_transfer["domains_exchangeable"]
        ),
    }
