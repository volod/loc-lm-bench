"""Exactness and maintained null calibration for the paired-reading decision rule."""

import json
from itertools import product
from pathlib import Path

import pytest

from llb.rag.fusion_evidence.randomization import (
    RANDOMIZATION_EXACT,
    RANDOMIZATION_MONTE_CARLO,
    paired_randomization,
    randomization_alpha,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "paired_randomization_null.json"
CONFIDENCE = 0.95
SEED = 13


def _brute_force_p(deltas: list[float]) -> float:
    magnitudes = [abs(value) for value in deltas if value]
    observed = sum(deltas)
    randomized = [
        sum(sign * value for sign, value in zip(signs, magnitudes))
        for signs in product((-1.0, 1.0), repeat=len(magnitudes))
    ]
    return sum(value >= observed - 1e-12 for value in randomized) / len(randomized)


@pytest.mark.parametrize(
    "deltas",
    [
        [1.0, 1.0, -1.0],
        [2.0, 0.5, -0.25, 0.0],
        [-1.0, -0.5, 0.25, 0.0],
        [0.0, 0.0],
    ],
)
def test_exact_randomization_p_matches_independent_brute_force(deltas: list[float]):
    result = paired_randomization(deltas, seed=SEED)
    assert result["method"] == RANDOMIZATION_EXACT
    assert result["p_value"] == pytest.approx(_brute_force_p(deltas))


def test_large_ledger_uses_reproducible_valid_monte_carlo_p():
    deltas = [1.0] * 12 + [-0.5] * 8
    first = paired_randomization(deltas, resamples=499, seed=SEED)
    second = paired_randomization(deltas, resamples=499, seed=SEED)
    assert first == second
    assert first["method"] == RANDOMIZATION_MONTE_CARLO
    assert first["samples"] == 499
    assert first["p_value"] >= 1 / 500


def test_shipped_rule_is_at_or_below_nominal_size_on_committed_null_fixtures():
    """Enumerate each fixture's whole null, then apply the exact shipped rule to every draw."""
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    alpha = randomization_alpha(CONFIDENCE)
    for fixture in fixtures:
        magnitudes = fixture["magnitudes"]
        assignments = list(product((-1.0, 1.0), repeat=len(magnitudes)))
        rejected = 0
        for signs in assignments:
            deltas = [sign * value for sign, value in zip(signs, magnitudes)]
            p_value = paired_randomization(deltas, seed=SEED)["p_value"]
            rejected += p_value <= alpha
        empirical_size = rejected / len(assignments)
        assert empirical_size <= alpha, (fixture["name"], empirical_size, alpha)
