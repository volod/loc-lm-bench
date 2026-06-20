from collections import Counter

import pytest

from llb.goldset.splits import assign_splits


def test_disjoint_and_deterministic():
    ids = [f"id{i}" for i in range(30)]
    a = assign_splits(ids, seed=7)
    b = assign_splits(ids, seed=7)
    assert a == b  # deterministic
    assert set(a) == set(ids)  # every id assigned exactly once
    assert set(a.values()) <= {"calibration", "tuning", "final"}


def test_counts_cover_total():
    ids = [f"id{i}" for i in range(100)]
    counts = Counter(assign_splits(ids).values())
    assert sum(counts.values()) == 100
    assert all(counts[s] > 0 for s in ("calibration", "tuning", "final"))


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits(["a"], ratios={"calibration": 0.5, "tuning": 0.4, "final": 0.4})
