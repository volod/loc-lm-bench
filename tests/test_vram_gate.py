import pytest

from llb.executor.vram import VramNotReclaimed, assert_reclaimed, wait_for_reclaim


def reader_from(seq):
    """A fake NVML reader yielding successive used-VRAM (MB) values."""
    values = iter(seq)
    return lambda: next(values)


def test_reclaim_succeeds_when_vram_returns_to_baseline():
    # used drops from 9000 to 2100 against a 2000 baseline (tolerance 512).
    result = wait_for_reclaim(2000, reader=reader_from([9000, 4000, 2100]), sleep=lambda _s: None)
    assert result["reclaimed"] is True
    assert result["residual_mb"] == 100


def test_reclaim_reports_residual_when_stuck():
    result = wait_for_reclaim(
        2000, reader=reader_from([9000] * 5), max_polls=5, sleep=lambda _s: None
    )
    assert result["reclaimed"] is False
    assert result["residual_mb"] == 7000


def test_assert_reclaimed_raises_on_leak():
    with pytest.raises(VramNotReclaimed):
        assert_reclaimed(2000, reader=reader_from([9000] * 3), max_polls=3, sleep=lambda _s: None)


def test_tolerated_baseline_shift_does_not_trip():
    # A small unrelated baseline rise (within tolerance) is reclaimed on the first poll.
    result = wait_for_reclaim(2000, reader=reader_from([2300]), sleep=lambda _s: None)
    assert result["reclaimed"] is True and result["polls"] == 1
