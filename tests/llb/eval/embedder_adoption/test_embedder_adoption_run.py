"""Focused tests split from ``test_embedder_adoption.py``."""

import json
from pathlib import Path

import pytest
from tests.llb.eval._embedder_adoption_helpers import (
    BASELINE,
    CANDIDATE,
    _gold_item,
    _lanes,
    _recording_lane,
    _write_goldset,
)

from llb.core.config import RunConfig
from llb.eval.embedder_adoption.models import EmbedderLane
from llb.eval.embedder_adoption.cells import build_cells
from llb.eval.embedder_adoption.run import run_adoption_bar_sweep


def test_every_cell_scores_the_same_items_under_both_encoders_and_persists(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset)
    seen: list[tuple[str, str, int, str | None]] = []

    run = run_adoption_bar_sweep(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        build_cells([10, 2], [None]),
        _lanes(tmp_path),
        out_dir=tmp_path / "adoption",
        resamples=50,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[0] for entry in seen] == [BASELINE, CANDIDATE] * 2
    assert [entry[2] for entry in seen] == [10, 10, 2, 2]
    assert {entry[1] for entry in seen} == {str(tmp_path / "e5"), str(tmp_path / "bge")}
    assert run.report["item_ids"] == ["q1", "q2"]
    assert [cell["label"] for cell in run.report["cells"]] == ["k10", "k2"]
    assert run.report["cells"][0]["lanes"][CANDIDATE]["run_dirs"] == [
        str(tmp_path / "run-eval" / f"embedder-adoption-k10-{CANDIDATE}-final")
    ]
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["grounding"] == "verified"
    assert persisted["metadata"]["split"] == "final"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Embedder adoption")


def test_the_sweep_compares_exactly_two_encoders(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset)
    with pytest.raises(ValueError, match="exactly two encoders"):
        run_adoption_bar_sweep(cfg, build_cells([10], [None]), _lanes(tmp_path)[:1])
    with pytest.raises(ValueError, match="must differ"):
        run_adoption_bar_sweep(
            cfg,
            build_cells([10], [None]),
            [EmbedderLane(BASELINE, tmp_path / "a"), EmbedderLane(BASELINE, tmp_path / "b")],
        )


def test_a_drafted_ledger_is_scorable_only_on_request_and_says_so_in_every_artifact(
    tmp_path: Path,
):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset, verified=False)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset)
    cells = build_cells([10], [None])
    with pytest.raises(SystemExit, match="no verified"):
        run_adoption_bar_sweep(cfg, cells, _lanes(tmp_path), run_lane=_recording_lane(tmp_path, []))

    run = run_adoption_bar_sweep(
        cfg,
        cells,
        _lanes(tmp_path),
        out_dir=tmp_path / "adoption",
        resamples=0,
        verified_only=False,
        run_lane=_recording_lane(tmp_path, []),
    )
    assert "grounding: `drafted`" in Path(run.paths["report"]).read_text(encoding="utf-8")


def test_several_splits_pool_into_one_compared_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    items = [_gold_item("q1"), _gold_item("q2")]
    items[1].split = "tuning"
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items),
        encoding="utf-8",
    )
    run = run_adoption_bar_sweep(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        build_cells([10], [None]),
        _lanes(tmp_path),
        splits=["final", "tuning"],
        out_dir=tmp_path / "adoption",
        resamples=0,
        run_lane=_recording_lane(tmp_path, []),
    )
    assert run.report["item_ids"] == ["q1", "q2"]
    assert len(run.report["cells"][0]["lanes"][BASELINE]["run_dirs"]) == 2
