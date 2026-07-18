"""Joint-search screen and finalist resume acceptance test."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec, ResolvedModel
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.joint_search.schedule import run_joint_search
from llb.optimize.objectives import TrialMetrics
from llb.optimize.tuning_space import FINAL_SPLIT


def test_resume_skips_completed_screen_and_finalist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Re-entry with the same run id skips completed screen and finalist cells."""
    specs: list[ModelSpec] = [
        {"name": name, "backend": "ollama", "source": f"{name}:tag"}
        for name in ("alpha", "bravo", "charlie", "delta")
    ]
    qualities = {"alpha": 0.4, "bravo": 0.9, "charlie": 0.7, "delta": 0.2}
    screen_calls: list[str] = []
    tune_calls: list[str] = []

    def fake_resolve_all(candidates, vram_mib, ram_mib, *, probes=None, **kwargs):
        del vram_mib, ram_mib, probes, kwargs
        return [
            ResolvedModel(
                name=candidate["name"],
                chosen_backend=candidate["backend"],
                chosen_source=candidate["source"],
                verdict="gpu",
                candidates=[],
                note="ok",
            )
            for candidate in candidates
        ]

    monkeypatch.setattr("llb.backends.resolver.resolve_all", fake_resolve_all)

    def screen_evaluate(config: RunConfig, limit: int | None) -> TrialMetrics:
        del limit
        name = config.model.split(":", 1)[0]
        screen_calls.append(name)
        return TrialMetrics(quality=qualities[name], latency_s=1.0)

    def tune_finalist(base: RunConfig, resolution: ResolvedModel, cell_dir: Path):
        del base, cell_dir
        name = resolution["name"]
        tune_calls.append(name)
        if name == "charlie" and tune_calls.count("charlie") == 1 and "bravo" in tune_calls:
            raise RuntimeError("simulated kill mid-finalist")
        return FinalistTuneResult(
            name=name,
            backend=resolution["chosen_backend"] or "ollama",
            source=resolution["chosen_source"] or name,
            study_name=f"joint-ci-resume-{name}",
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
    with pytest.raises(RuntimeError, match="simulated kill"):
        run_joint_search(
            base,
            specs,
            n_trials=5,
            run_id="ci-resume",
            screen_limit=4,
            min_finalists=2,
            eta=2,
            screen_evaluate=screen_evaluate,
            tune_finalist=tune_finalist,
            isolate=False,
        )

    assert sorted(screen_calls) == ["alpha", "bravo", "charlie", "delta"]
    assert tune_calls == ["bravo", "charlie"]
    finalists_dir = tmp_path / "joint-search" / "ci-resume" / "finalists"
    assert (finalists_dir / "bravo" / "result.json").is_file()
    assert not (finalists_dir / "charlie" / "result.json").is_file()

    screen_before = list(screen_calls)
    tune_before = list(tune_calls)
    result = run_joint_search(
        base,
        specs,
        n_trials=5,
        run_id="ci-resume",
        screen_limit=4,
        min_finalists=2,
        eta=2,
        screen_evaluate=screen_evaluate,
        tune_finalist=tune_finalist,
        isolate=False,
    )
    assert screen_calls == screen_before
    assert tune_calls == tune_before + ["charlie"]
    assert set(result.ledger.finalists) == {"bravo", "charlie"}
    assert result.recommended is not None
    assert result.recommended["model"] == "bravo"
    board = json.loads((result.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert {entry["model"] for entry in board["entries"]} == {"bravo", "charlie"}
