"""Westfall-Young step-down max-statistic family adjustment."""

from itertools import product

import pytest

from llb.rag.fusion_evidence.selection import selection_adjustment


def _statistic(values: list[float]) -> float:
    mean = sum(values) / len(values)
    centered = sum((value - mean) ** 2 for value in values)
    if centered == 0.0:
        return float("inf") if mean > 0.0 else float("-inf") if mean < 0.0 else 0.0
    return sum(values) / ((len(values) * centered / (len(values) - 1)) ** 0.5)


def _brute_force(hypotheses: dict[str, list[float]]) -> dict[str, float]:
    keys = list(hypotheses)
    observed = {key: _statistic(hypotheses[key]) for key in keys}
    order = sorted(keys, key=lambda key: (-observed[key], key))
    null = {key: [] for key in keys}
    for signs in product((-1.0, 1.0), repeat=len(next(iter(hypotheses.values())))):
        for key in keys:
            null[key].append(
                _statistic([sign * value for sign, value in zip(signs, hypotheses[key])])
            )
    adjusted: dict[str, float] = {}
    running = 0.0
    for position, key in enumerate(order):
        remaining = order[position:]
        extreme = sum(
            max(null[member][draw] for member in remaining) >= observed[key]
            for draw in range(len(null[key]))
        )
        running = max(running, extreme / len(null[key]))
        adjusted[key] = running
    return adjusted


def test_step_down_adjustment_matches_an_independent_brute_force_family():
    hypotheses = {
        "strong": [2.0, 1.0, 0.5, -0.25],
        "correlated": [1.0, 0.5, 0.25, -0.125],
        "other": [0.25, -1.0, 0.5, 0.0],
    }
    result = selection_adjustment(hypotheses, seed=13)

    expected = _brute_force(hypotheses)
    assert result["randomization_method"] == "exact"
    assert result["samples"] == 16
    assert result["family_size"] == 3
    for key, value in expected.items():
        assert result["p_values"][key]["adjusted_p"] == pytest.approx(value)
        assert result["p_values"][key]["adjusted_p"] >= result["p_values"][key]["unadjusted_p"]


def test_jointly_zero_item_columns_do_not_make_a_small_family_monte_carlo():
    result = selection_adjustment(
        {"a": [1.0, -0.5, 0.0] + [0.0] * 20, "b": [0.5, 0.25, 0.0] + [0.0] * 20},
        seed=13,
    )
    assert result["randomization_method"] == "exact"
    assert result["samples"] == 4
    assert result["items"] == 23


def test_monte_carlo_family_is_reproducible_and_uses_the_plus_one_correction():
    hypotheses = {
        "a": [float(index % 5 - 2) for index in range(20)],
        "b": [float(index % 7 - 3) for index in range(20)],
    }
    first = selection_adjustment(hypotheses, resamples=99, seed=7, exact_limit=3)
    second = selection_adjustment(hypotheses, resamples=99, seed=7, exact_limit=3)
    assert first == second
    assert first["randomization_method"] == "monte_carlo"
    assert all(entry["adjusted_p"] >= 0.01 for entry in first["p_values"].values())


def test_family_requires_aligned_finite_vectors():
    with pytest.raises(ValueError, match="same aligned"):
        selection_adjustment({"a": [1.0], "b": [1.0, 2.0]}, seed=13)
    with pytest.raises(ValueError, match="finite"):
        selection_adjustment({"a": [float("nan")]}, seed=13)
