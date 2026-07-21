"""Tests for joint model + RAG-config successive-halving search."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec, ResolvedModel
from llb.optimize.joint_search.halving import finalize_ledger
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.joint_search.schedule import run_joint_search
from llb.optimize.joint_search.schedule_steps import partition_resolved
from llb.optimize.objectives import TrialMetrics
from llb.optimize.tuning_space import FINAL_SPLIT, TUNING_SPLIT


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


def test_partition_resolved_skips_missing_local_runtime(tmp_path: Path, monkeypatch) -> None:
    row = ResolvedModel(
        name="gguf-only",
        chosen_backend="llamacpp",
        chosen_source="hf.co/org/model-GGUF:Q4_K_M",
        verdict="offload",
        candidates=[],
        note="ok",
    )
    monkeypatch.setattr("llb.backends.readiness.shutil.which", lambda _name: None)

    runnable, skipped = partition_resolved([row], data_dir=tmp_path)

    assert runnable == []
    assert skipped == [
        {"name": "gguf-only", "reason": "llama-server not found (run make build-llamacpp)"}
    ]


def _name_from_source(source: str) -> str:
    return source.split(":", 1)[0]
