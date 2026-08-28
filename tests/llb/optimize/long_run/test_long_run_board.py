"""The confidence-aware board: intervals, paired deltas, the frontier, and the verdict."""

import json
from pathlib import Path
from typing import Any

from llb.optimize.joint_search.long_run.public_tracks import summarize
from llb.optimize.joint_search.long_run.uncertainty import (
    BoardRow,
    pareto_frontier,
    read_board_rows,
    read_uncertainty,
)
from llb.optimize.joint_search.long_run.verdict import (
    DECISION_ADOPT,
    DECISION_RETAIN,
    DECISION_UNDECIDED,
    decide,
)
from llb.optimize.tuning_space import FINAL_SPLIT

NO_PUBLIC: dict[str, Any] = {"reports": {}, "tracks": [], "comparable": True, "complete": {}}


def _row(model: str, pick: str, quality: list[float], latency: list[float]) -> BoardRow:
    return BoardRow(model=model, pick=pick, backend="ollama", quality=quality, latency=latency)


def _clear_win(n: int = 40) -> tuple[list[float], list[float]]:
    """A candidate that beats the baseline on every item -- a delta no resample can straddle."""
    return [1.0] * n, [0.0] * n


def test_a_point_lead_that_no_resample_supports_retains_the_incumbent():
    """One item's worth of lead over 40 items is exactly the reversal this lane refuses."""
    baseline = [1.0 if i % 2 else 0.0 for i in range(40)]
    candidate = list(baseline)
    candidate[0] = 1.0
    rows = [
        _row("incumbent", "best_quality", baseline, [1.0] * 40),
        _row("challenger", "best_quality", candidate, [1.0] * 40),
    ]
    uncertainty = read_uncertainty(rows, incumbent="incumbent", resamples=200)
    verdict = decide(uncertainty, incumbent="incumbent", public=NO_PUBLIC)
    assert verdict.decision == DECISION_RETAIN
    assert verdict.model == "incumbent"
    assert "does not" not in verdict.reason  # the retain reason names the item set, not a hedge
    assert f"{uncertainty.n_items}-item" in verdict.reason


def test_a_separated_candidate_is_adopted_and_the_delta_is_quoted():
    winner, loser = _clear_win()
    rows = [
        _row("incumbent", "best_quality", loser, [1.0] * 40),
        _row("challenger", "best_quality", winner, [1.0] * 40),
    ]
    uncertainty = read_uncertainty(rows, incumbent="incumbent", resamples=200)
    verdict = decide(uncertainty, incumbent="incumbent", public=NO_PUBLIC)
    assert verdict.decision == DECISION_ADOPT
    assert verdict.model == "challenger"
    assert verdict.row == "challenger::best_quality"
    assert "+1.000" in verdict.reason


def test_an_unscored_incumbent_leaves_the_decision_undecided():
    rows = [_row("challenger", "best_quality", [1.0] * 10, [1.0] * 10)]
    uncertainty = read_uncertainty(rows, incumbent="incumbent", resamples=50)
    verdict = decide(uncertainty, incumbent="incumbent", public=NO_PUBLIC)
    assert verdict.decision == DECISION_UNDECIDED
    assert "was not scored on the held-out split" in verdict.reason


def test_the_verdict_states_the_quality_latency_tradeoff_when_the_two_disagree():
    """A slower quality leader must say so: the frontier is half of what an operator acts on."""
    winner, loser = _clear_win()
    rows = [
        _row("incumbent", "best_quality", loser, [0.5] * 40),
        _row("challenger", "best_quality", winner, [4.0] * 40),
    ]
    uncertainty = read_uncertainty(rows, incumbent="incumbent", resamples=200)
    verdict = decide(uncertainty, incumbent="incumbent", public=NO_PUBLIC)
    assert verdict.quality_leader == "challenger::best_quality"
    assert verdict.latency_leader == "incumbent::best_quality"
    assert "trading quality for latency" in verdict.tradeoff
    assert set(verdict.frontier) == {"challenger::best_quality", "incumbent::best_quality"}


def test_a_dominated_row_is_off_the_frontier():
    """Worse quality AND slower is dominated; equal-on-both keeps both rows."""
    frontier = pareto_frontier(
        [
            _row("a", "q", [1.0], [1.0]),
            _row("b", "q", [0.5], [2.0]),
            _row("c", "q", [0.5], [0.5]),
        ]
    )
    assert set(frontier) == {"a::q", "c::q"}


def test_a_missing_public_screen_qualifies_the_adoption_sentence():
    winner, loser = _clear_win()
    rows = [
        _row("incumbent", "best_quality", loser, [1.0] * 40),
        _row("challenger", "best_quality", winner, [1.0] * 40),
    ]
    uncertainty = read_uncertainty(rows, incumbent="incumbent", resamples=200)
    verdict = decide(uncertainty, incumbent="incumbent", public=summarize({}))
    assert verdict.decision == DECISION_ADOPT
    assert "no public Ukrainian screen" in verdict.reason


def test_a_partial_public_screen_qualifies_the_adoption_sentence():
    winner, loser = _clear_win()
    rows = [
        _row("incumbent", "best_quality", loser, [1.0] * 40),
        _row("challenger", "best_quality", winner, [1.0] * 40),
    ]
    public = summarize(
        {
            "challenger": {
                "model": "challenger:tag",
                "backend": "ollama",
                "track": "generation",
                "requested_tasks": ["t1", "t2"],
                "results": [{"task": "t1", "metric": "acc", "score": 0.5}],
                "covered": ["t1"],
                "missing": ["t2"],
                "complete": False,
            }
        }
    )
    uncertainty = read_uncertainty(rows, incumbent="incumbent", resamples=200)
    verdict = decide(uncertainty, incumbent="incumbent", public=public)
    assert verdict.decision == DECISION_ADOPT
    assert "PARTIAL" in verdict.reason


def test_board_rows_are_read_back_per_case_and_paired_on_the_shared_items(tmp_path: Path):
    alpha = _write_scores(
        tmp_path / "alpha", {"i1": (1.0, 0.5), "i2": (1.0, 0.5), "i3": (0.0, 1.0)}
    )
    bravo = _write_scores(tmp_path / "bravo", {"i1": (0.0, 2.0), "i2": (0.0, 2.0)})
    entries = [
        {"model": "alpha", "pick": "best_quality", "backend": "ollama", "split": FINAL_SPLIT},
        {"model": "bravo", "pick": "best_quality", "backend": "ollama", "split": FINAL_SPLIT},
    ]
    finals = {
        "alpha": {"best_quality": {"paths": {"scores": str(alpha)}}},
        "bravo": {"best_quality": {"paths": {"scores": str(bravo)}}},
    }
    rows, unreadable = read_board_rows(entries, finals)  # type: ignore[arg-type]
    assert not unreadable
    # `i3` is scored by alpha alone, so the paired item set is the two both scored.
    assert [len(row.quality) for row in rows] == [2, 2]
    uncertainty = read_uncertainty(rows, incumbent="bravo", resamples=100)
    assert uncertainty.n_items == 2
    assert uncertainty.baseline == "bravo::best_quality"
    assert uncertainty.paired["alpha::best_quality"]["wins"] == 2


def test_a_row_with_no_scores_file_is_reported_not_silently_compared(tmp_path: Path):
    alpha = _write_scores(tmp_path / "alpha", {"i1": (1.0, 0.5)})
    entries = [
        {"model": "alpha", "pick": "best_quality", "backend": "ollama"},
        {"model": "bravo", "pick": "best_quality", "backend": "ollama"},
    ]
    finals = {
        "alpha": {"best_quality": {"paths": {"scores": str(alpha)}}},
        "bravo": {"best_quality": {"paths": {}}},
    }
    rows, unreadable = read_board_rows(entries, finals)  # type: ignore[arg-type]
    assert [row.key for row in rows] == ["alpha::best_quality"]
    assert unreadable == [{"row": "bravo::best_quality", "reason": "no scores.jsonl"}]
    assert (
        read_uncertainty(rows, incumbent="alpha", unreadable=unreadable).to_dict()[
            "unreadable_rows"
        ]
        == unreadable
    )


def _write_scores(run_dir: Path, cases: dict[str, tuple[float, float]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "scores.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"item_id": item, "objective_score": quality, "latency_s": latency})
            for item, (quality, latency) in cases.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return path
