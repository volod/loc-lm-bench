"""The whole confirmation schedule on fakes: derived screen, blocks, verdict, artifact."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec, ResolvedModel
from llb.core.contracts.screening import ScreenReport
from llb.optimize.joint_search.long_run.plan import declare_plan
from llb.optimize.joint_search.long_run.run import run_long_run
from llb.optimize.joint_search.long_run.stage import FinalistCell, LongRunStage
from llb.optimize.objectives import GoalPick, ParetoPoint, TrialMetrics
from llb.optimize.tuner_models import MultiObjectiveResult
from llb.optimize.tuning_space import FINAL_SPLIT, TUNING_SPLIT

DELTAS = [0.6, -0.5, 0.4, -0.3, 0.7, -0.6, 0.2, -0.4, 0.5, -0.2]
SPECS: list[ModelSpec] = [
    {"name": "alpha", "backend": "ollama", "source": "alpha:tag"},
    {"name": "bravo", "backend": "ollama", "source": "bravo:tag"},
    {"name": "charlie", "backend": "ollama", "source": "charlie:tag"},
    {"name": "delta", "backend": "ollama", "source": "delta:tag"},
]
SCREEN_QUALITY = {"alpha": 0.9, "bravo": 0.7, "charlie": 0.4, "delta": 0.2}
# alpha wins every held-out item, bravo none: a delta the bootstrap cannot straddle.
FINAL_QUALITY = {"alpha": 1.0, "bravo": 0.0}
FINAL_LATENCY = {"alpha": 2.0, "bravo": 0.5}


@pytest.fixture
def fake_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _plan(tmp_path: Path, *, available_n: int = 82, budget: int = 30, block: int = 5):
    return declare_plan(
        tmp_path / "reference.json",
        DELTAS,
        minimum_detectable_gain=0.25,
        available_n=available_n,
        trial_budget=budget,
        trial_block=block,
        stability_blocks=2,
        stability_agreement=1.0,
        selector={"lane": "test", "candidate": "a", "baseline": "b", "metric": "q"},
    )


def _tune_block(cell: FinalistCell, target: int) -> MultiObjectiveResult:
    """A study whose front is one point: the finalist's stable tuning-split quality."""
    name = cell.name
    point = ParetoPoint(
        number=target,
        quality=SCREEN_QUALITY[name],
        latency_s=1.0,
        cost_usd=0.0,
        throughput=1.0,
        overrides={"top_k": 5},
    )
    return MultiObjectiveResult(
        study_name=cell.study_name,
        storage=None,
        objectives=("quality", "latency"),
        n_trials=target,
        n_complete=target,
        n_pruned=0,
        front=[point],
        picks=[GoalPick("best_quality", point)],
        report_dir=cell.cell_dir,
    )


def _final_runner(scores_root: Path):
    """A held-out runner that writes a real `scores.jsonl` so the board is re-read per case."""

    def run(config: RunConfig):
        name = "alpha" if "alpha" in config.model else "bravo"
        run_dir = scores_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "scores.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "item_id": f"i{i}",
                        "objective_score": FINAL_QUALITY[name],
                        "latency_s": FINAL_LATENCY[name],
                    }
                )
                for i in range(40)
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "rows": [{"model": name, "quality": FINAL_QUALITY[name]}],
            "metrics": {"objective_score": FINAL_QUALITY[name]},
            "manifest": {"split": FINAL_SPLIT},
            "table": "ok",
            "retrieval": {},
            "paths": {"scores": str(path)},
            "telemetry": None,
            "run_timestamp": "t",
        }

    return run


def _screen_runner(complete: bool = True):
    def run(source: str, backend: str, out_dir: Path) -> ScreenReport:
        del out_dir
        return {
            "model": source,
            "backend": backend,
            "track": "generation",
            "requested_tasks": ["global_piqa_prompted_ukr_cyrl"],
            "results": [{"task": "global_piqa_prompted_ukr_cyrl", "metric": "acc", "score": 0.61}],
            "covered": ["global_piqa_prompted_ukr_cyrl"] if complete else [],
            "missing": [] if complete else ["global_piqa_prompted_ukr_cyrl"],
            "complete": complete,
        }

    return run


def _run(tmp_path: Path, *, plan, screen_runner, run_id: str = "ci-long-run"):
    screen_limits: list[int] = []

    def screen_evaluate(config: RunConfig, limit: int | None) -> TrialMetrics:
        assert limit is not None
        screen_limits.append(limit)
        name = next(n for n in SCREEN_QUALITY if n in config.model)
        return TrialMetrics(quality=SCREEN_QUALITY[name], latency_s=1.0)

    stage = LongRunStage(
        plan=plan,
        objectives=("quality", "latency"),
        isolate=False,
        tune_block=_tune_block,
        final_runner=_final_runner(tmp_path / "runs"),
    )
    result = run_long_run(
        RunConfig(data_dir=tmp_path),
        SPECS,
        plan=plan,
        incumbent="bravo",
        run_id=run_id,
        min_finalists=2,
        isolate=False,
        screen_evaluate=screen_evaluate,
        stage=stage,
        screen_runner=screen_runner,
    )
    return result, screen_limits


def test_the_confirmation_screens_at_the_derived_size_and_scores_the_full_split(
    tmp_path: Path, fake_resolver: None
):
    """The screen cap is the power-derived count, and no case limit reaches the held-out runs."""
    plan = _plan(tmp_path)
    result, screen_limits = _run(tmp_path, plan=plan, screen_runner=_screen_runner())
    assert screen_limits == [plan.screen.applied_n] * len(SPECS)
    assert plan.screen.applied_n == plan.screen.required_n
    manifest = json.loads((result.search.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_limit"] is None
    assert manifest["mode"] == "joint-search-long-run"
    assert manifest["incumbent"] == "bravo"
    assert manifest["screen_limit"] == plan.screen.applied_n


def test_no_final_split_score_reaches_the_screen_or_the_stopping_rule(
    tmp_path: Path, fake_resolver: None
):
    """Eliminations and every block ranking are tuning-split quantities; the board is final."""
    result, _ = _run(tmp_path, plan=_plan(tmp_path), screen_runner=_screen_runner())
    assert result.search.ledger.split == TUNING_SPLIT
    assert all(round_.split == TUNING_SPLIT for round_ in result.search.ledger.rounds)
    # The stopping rule saw the tuning-split screen values, never the held-out 1.0 / 0.0.
    for block in result.trail["blocks"]:
        assert block["objective"] == {
            name: SCREEN_QUALITY[name] for name in result.search.ledger.finalists
        }
    board = json.loads((result.search.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert board["split"] == FINAL_SPLIT
    assert all(entry["split"] == FINAL_SPLIT for entry in board["entries"])


def test_the_scoreboard_says_its_recommended_row_is_only_the_point_estimate(
    tmp_path: Path, fake_resolver: None
):
    """An operator opening `scoreboard.md` alone is pointed at the verdict that supersedes it."""
    result, _ = _run(tmp_path, plan=_plan(tmp_path), screen_runner=_screen_runner())
    board = json.loads((result.search.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert "point-estimate argmax" in board["note"]
    assert "long_run.md" in (result.search.run_dir / "scoreboard.md").read_text(encoding="utf-8")


def test_the_search_stops_on_stability_and_records_what_it_consumed(
    tmp_path: Path, fake_resolver: None
):
    result, _ = _run(tmp_path, plan=_plan(tmp_path), screen_runner=_screen_runner())
    assert result.trail["stopped_by"] == "ranking-stability"
    assert result.trail["trials_per_finalist"] == 15
    assert result.trail["consumed_total"] == 30
    assert result.trail["budget_exhausted"] is False


def test_the_artifact_carries_the_declaration_the_trail_and_the_verdict(
    tmp_path: Path, fake_resolver: None
):
    """Everything the acceptance gate asks for lands in one readable record."""
    result, _ = _run(tmp_path, plan=_plan(tmp_path), screen_runner=_screen_runner())
    payload = json.loads(result.paths["json"].read_text(encoding="utf-8"))
    declaration = payload["predeclaration"]
    assert declaration["minimum_detectable_gain"] == 0.25
    assert declaration["target_power"] == pytest.approx(0.8)
    assert declaration["screen"]["required_n"] > 0
    assert declaration["power"]["method"] == "paired-normal-approximation"
    assert "consecutive block transitions" in declaration["stopping_rule"]
    assert payload["search"]["stopped_by"] == "ranking-stability"
    assert payload["search"]["consumed_total"] == 30
    assert payload["final"]["split"] == FINAL_SPLIT
    assert payload["final"]["uncertainty"]["n_items"] == 40
    assert payload["verdict"]["decision"] == "adopt"
    assert payload["verdict"]["model"] == "alpha"
    assert payload["public_screen"]["complete"] == {"alpha": True, "bravo": True}
    realized = payload["final"]["power"]
    assert realized["realized_n"] == 40
    assert realized["resolution"] == "separated"
    assert realized["direction"] == "alpha::best_quality"
    markdown = result.paths["markdown"].read_text(encoding="utf-8")
    assert "Verdict: ADOPT" in markdown
    assert "Public Ukrainian screen" in markdown
    assert "Paired power (declared)" in markdown
    assert "Paired power (realized on the held-out split)" in markdown


def test_a_failing_public_screen_qualifies_the_verdict_instead_of_losing_the_board(
    tmp_path: Path, fake_resolver: None
):
    def explode(source: str, backend: str, out_dir: Path) -> ScreenReport:
        del backend, out_dir
        raise RuntimeError(f"lm_eval missing for {source}")

    result, _ = _run(tmp_path, plan=_plan(tmp_path), screen_runner=explode)
    assert [f["model"] for f in result.public["failures"]] == ["alpha", "bravo"]
    assert result.verdict.decision == "adopt"
    assert "no public Ukrainian screen" in result.verdict.reason
    assert result.paths["json"].is_file()


def test_a_tuning_split_below_the_derived_size_is_reported_not_rounded_away(
    tmp_path: Path, fake_resolver: None
):
    plan = _plan(tmp_path, available_n=6)
    result, screen_limits = _run(tmp_path, plan=plan, screen_runner=_screen_runner())
    assert screen_limits == [6] * len(SPECS)
    payload = json.loads(result.paths["json"].read_text(encoding="utf-8"))
    screen = payload["predeclaration"]["screen"]
    assert screen["applied_n"] == 6
    assert screen["required_n"] > 6
    assert screen["binding"] == "tuning-split-exhausted"
    assert screen["satisfied"] is False


def test_a_resumed_finalist_is_reloaded_rather_than_retuned(tmp_path: Path, fake_resolver: None):
    """Re-entry with the same run id reuses finished finalists and still writes the verdict."""
    first, _ = _run(tmp_path, plan=_plan(tmp_path), screen_runner=_screen_runner())
    assert first.trail["blocks"]
    second, _ = _run(tmp_path, plan=_plan(tmp_path), screen_runner=_screen_runner())
    assert second.trail["blocks"] == []
    assert second.trail["consumed_total"] == 0
    assert {f.name for f in second.search.finalists} == {"alpha", "bravo"}
    assert second.verdict.decision == "adopt"
