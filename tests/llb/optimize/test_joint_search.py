"""Tests for joint model + RAG-config successive-halving search."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec, ResolvedModel
from llb.optimize.joint_search.halving import (
    ScreenScore,
    build_halving_round,
    finalize_ledger,
    keep_count,
    partition_survivors,
    rank_scores,
    screen_limit_for_round,
)
from llb.optimize.joint_search.report import assert_final_split, write_scoreboard
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.joint_search.schedule import run_joint_search
from llb.optimize.objectives import TrialMetrics
from llb.optimize.tuning_space import FINAL_SPLIT, TUNING_SPLIT


def test_rank_scores_quality_then_name():
    scores = [
        ScreenScore("b", 0.5),
        ScreenScore("a", 0.5),
        ScreenScore("c", 0.9),
    ]
    ranked = rank_scores(scores)
    assert [s.name for s in ranked] == ["c", "a", "b"]


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


def test_run_joint_search_fake_hooks_no_tuning_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """CI schedule: fake resolve + screen + tune; ledger tuning-only, scoreboard final-only."""
    specs: list[ModelSpec] = [
        {"name": "alpha", "backend": "ollama", "source": "alpha:tag"},
        {"name": "bravo", "backend": "ollama", "source": "bravo:tag"},
        {"name": "charlie", "backend": "ollama", "source": "charlie:tag"},
        {"name": "delta", "backend": "ollama", "source": "delta:tag"},
    ]
    qualities = {"alpha": 0.4, "bravo": 0.9, "charlie": 0.7, "delta": 0.2}
    screen_splits: list[str] = []
    screen_limits: list[int] = []

    def fake_resolve_all(candidates, vram_mib, ram_mib, *, probes=None, **kwargs):
        del vram_mib, ram_mib, probes, kwargs
        return [
            ResolvedModel(
                name=c["name"],
                chosen_backend=c["backend"],
                chosen_source=c["source"],
                verdict="gpu",
                candidates=[],
                note="ok",
            )
            for c in candidates
        ]

    monkeypatch.setattr("llb.backends.resolver.resolve_all", fake_resolve_all)

    def screen_evaluate(config: RunConfig, limit: int | None) -> TrialMetrics:
        screen_splits.append(TUNING_SPLIT)  # schedule contract: always tuning
        assert limit is not None
        screen_limits.append(limit)
        name = _name_from_source(config.model)
        return TrialMetrics(quality=qualities[name], latency_s=1.0)

    def tune_finalist(base: RunConfig, resolution: ResolvedModel, cell_dir: Path):
        del base, cell_dir
        name = resolution["name"]
        return FinalistTuneResult(
            name=name,
            backend=resolution["chosen_backend"] or "ollama",
            source=resolution["chosen_source"] or name,
            study_name=f"joint-fake-{name}",
            overrides_by_pick={"best_quality": {"top_k": 5}},
            finals={
                "best_quality": {
                    "rows": [{"model": name, "quality": qualities[name] + 0.05}],
                    "metrics": {"objective_score": qualities[name] + 0.05},
                    "manifest": {"split": FINAL_SPLIT},
                    "table": "ok",
                    "retrieval": {},
                    "paths": {},
                    "telemetry": None,
                    "run_timestamp": "t",
                }
            },
        )

    base = RunConfig(data_dir=tmp_path)
    result = run_joint_search(
        base,
        specs,
        n_trials=5,
        run_id="ci-joint",
        screen_limit=4,
        min_finalists=2,
        eta=2,
        screen_evaluate=screen_evaluate,
        tune_finalist=tune_finalist,
        isolate=False,
    )
    assert result.ledger.split == TUNING_SPLIT
    assert all(r.split == TUNING_SPLIT for r in result.ledger.rounds)
    assert set(result.ledger.finalists) == {"bravo", "charlie"}
    assert screen_splits and all(s == TUNING_SPLIT for s in screen_splits)
    # Round 0 limit=4 for each of 4 candidates; keep 2 and stop (no round 1).
    assert screen_limits == [4, 4, 4, 4]
    assert result.recommended is not None
    assert result.recommended["model"] == "bravo"
    assert result.recommended["split"] == FINAL_SPLIT

    ledger = json.loads((result.run_dir / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["split"] == TUNING_SPLIT
    assert ledger["finalists"] == ["bravo", "charlie"]
    eliminated = {name for r in ledger["rounds"] for name in r["eliminated"]}
    assert eliminated == {"alpha", "delta"}

    board = json.loads((result.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert board["split"] == FINAL_SPLIT
    assert all(e["split"] == FINAL_SPLIT for e in board["entries"])
    assert (result.run_dir / "scoreboard.md").is_file()


def test_halving_two_rounds_increases_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Eight candidates need two rounds; round-1 screen_limit grows by eta."""
    specs: list[ModelSpec] = [
        {"name": f"m{i}", "backend": "ollama", "source": f"m{i}:tag"} for i in range(8)
    ]
    # Higher index -> higher quality so keep order is deterministic.
    qualities = {f"m{i}": i / 10.0 for i in range(8)}
    limits: list[int] = []

    def fake_resolve_all(candidates, vram_mib, ram_mib, *, probes=None, **kwargs):
        del vram_mib, ram_mib, probes, kwargs
        return [
            ResolvedModel(
                name=c["name"],
                chosen_backend="ollama",
                chosen_source=c["source"],
                verdict="gpu",
                candidates=[],
                note="ok",
            )
            for c in candidates
        ]

    monkeypatch.setattr("llb.backends.resolver.resolve_all", fake_resolve_all)

    def screen_evaluate(config: RunConfig, limit: int | None) -> TrialMetrics:
        assert limit is not None
        limits.append(limit)
        name = _name_from_source(config.model)
        return TrialMetrics(quality=qualities[name], latency_s=1.0)

    def tune_finalist(base: RunConfig, resolution: ResolvedModel, cell_dir: Path):
        del base, cell_dir
        name = resolution["name"]
        return FinalistTuneResult(
            name=name,
            backend="ollama",
            source=resolution["chosen_source"] or name,
            study_name=f"j-{name}",
            overrides_by_pick={"best_quality": {}},
            finals={
                "best_quality": {
                    "rows": [{"model": name, "quality": 0.5}],
                    "metrics": {"objective_score": 0.5},
                    "manifest": {"split": FINAL_SPLIT},
                    "table": "ok",
                    "retrieval": {},
                    "paths": {},
                    "telemetry": None,
                    "run_timestamp": "t",
                }
            },
        )

    result = run_joint_search(
        RunConfig(data_dir=tmp_path),
        specs,
        n_trials=3,
        run_id="two-round",
        screen_limit=5,
        min_finalists=2,
        eta=2,
        screen_evaluate=screen_evaluate,
        tune_finalist=tune_finalist,
        isolate=False,
    )
    assert len(result.ledger.rounds) == 2
    assert result.ledger.rounds[0].case_limit == 5
    assert result.ledger.rounds[1].case_limit == 10
    # 8 screen calls in round 0 + 4 in round 1.
    assert limits.count(5) == 8
    assert limits.count(10) == 4
    assert set(result.ledger.finalists) == {"m6", "m7"}


def test_finalize_ledger_empty():
    ledger = finalize_ledger([])
    assert ledger.finalists == ()
    assert ledger.split == TUNING_SPLIT


def _name_from_source(source: str) -> str:
    return source.split(":", 1)[0]
