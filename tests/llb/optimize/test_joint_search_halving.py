"""Successive-halving and final-split fence unit tests."""

from pathlib import Path

import pytest

from llb.optimize.joint_search.halving import (
    ScreenScore,
    build_halving_round,
    keep_count,
    partition_survivors,
    rank_scores,
    screen_limit_for_round,
)
from llb.optimize.joint_search.report import assert_final_split, write_scoreboard
from llb.optimize.tuning_space import FINAL_SPLIT, TUNING_SPLIT


def test_rank_scores_quality_then_name():
    scores = [ScreenScore("b", 0.5), ScreenScore("a", 0.5), ScreenScore("c", 0.9)]
    assert [score.name for score in rank_scores(scores)] == ["c", "a", "b"]


def test_keep_count_halves_with_floor():
    assert keep_count(8, eta=2, min_keep=2) == 4
    assert keep_count(3, eta=2, min_keep=2) == 2
    assert keep_count(2, eta=2, min_keep=2) == 2


def test_partition_survivors_eliminates_bottom_half():
    scores = [
        ScreenScore("m1", 0.9),
        ScreenScore("m2", 0.8),
        ScreenScore("m3", 0.7),
        ScreenScore("m4", 0.1),
    ]
    kept, eliminated = partition_survivors(scores, eta=2, min_keep=2)
    assert kept == ["m1", "m2"]
    assert eliminated == ["m3", "m4"]


def test_build_halving_round_rejects_final_split():
    with pytest.raises(ValueError, match="tuning"):
        build_halving_round(
            [ScreenScore("m", 0.5)],
            round_index=0,
            case_limit=8,
            split=FINAL_SPLIT,
        )


def test_screen_limit_grows_by_eta():
    assert screen_limit_for_round(8, 0, eta=2) == 8
    assert screen_limit_for_round(8, 1, eta=2) == 16
    assert screen_limit_for_round(8, 2, eta=2) == 32


def test_write_scoreboard_rejects_tuning_leak(tmp_path: Path):
    with pytest.raises(ValueError, match="final"):
        write_scoreboard(
            tmp_path,
            run_id="r1",
            entries=[
                {
                    "model": "m",
                    "pick": "best_quality",
                    "quality": 0.5,
                    "split": TUNING_SPLIT,
                }
            ],
        )
    with pytest.raises(ValueError, match="final"):
        assert_final_split({"model": "m", "split": TUNING_SPLIT})
