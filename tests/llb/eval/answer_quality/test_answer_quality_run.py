"""Focused tests split from ``test_answer_quality.py``."""

import json
from pathlib import Path

import pytest
from tests.llb.eval._answer_quality_helpers import (
    FUSED,
    VECTOR,
    _gold_item,
    _recording_lane,
    _write_bundle,
)

from llb.core.config import RunConfig
from llb.eval.answer_quality import (
    parse_lanes,
    run_answer_quality,
)


def test_every_lane_scores_the_same_selected_items_and_the_comparison_persists(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        parse_lanes(f"{VECTOR},{FUSED}"),
        out_dir=tmp_path / "answer-quality",
        resamples=50,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[1] for entry in seen] == ["faiss", "fused"]
    assert {entry[2] for entry in seen} == {("q1", "q2")}
    assert run.report["item_ids"] == ["q1", "q2"]
    assert run.report["lanes"][FUSED]["run_dirs"] == [
        str(tmp_path / "run-eval" / f"answer-quality-{FUSED}-final")
    ]
    assert run.report["verdict"]["focus_n"] == 1
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["split"] == "final"
    assert persisted["metadata"]["grounding"] == "verified"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Multi-hop")


def test_a_drafted_ledger_is_scorable_only_on_request_and_says_so_in_every_artifact(
    tmp_path: Path,
):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset, verified=False)
    lanes = parse_lanes(f"{VECTOR},{FUSED}")
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset)
    with pytest.raises(SystemExit, match="no verified"):
        run_answer_quality(cfg, lanes, run_lane=_recording_lane(tmp_path, []))

    run = run_answer_quality(
        cfg,
        lanes,
        out_dir=tmp_path / "answer-quality",
        resamples=0,
        verified_only=False,
        run_lane=_recording_lane(tmp_path, []),
    )
    assert run.report["item_ids"] == ["q1", "q2"]
    assert "grounding: `drafted`" in Path(run.paths["report"]).read_text(encoding="utf-8")


def test_a_single_lane_is_not_a_comparison(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(ValueError, match="at least one candidate lane"):
        run_answer_quality(RunConfig(data_dir=tmp_path, goldset_path=goldset), parse_lanes(VECTOR))


def test_several_splits_pool_into_one_compared_item_set(tmp_path: Path):
    """One ordinary run bundle per (lane, split); the comparison covers the pooled ledger."""
    goldset = tmp_path / "goldset.jsonl"
    items = [_gold_item("q1"), _gold_item("q2")]
    items[1].split = "tuning"
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8"
    )
    (goldset.parent / "needle_items.jsonl").write_text(
        '{"id": "q1", "question_type": "multi-hop"}\n{"id": "q2", "question_type": "multi-hop"}\n',
        encoding="utf-8",
    )
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        parse_lanes(f"{VECTOR},{FUSED}"),
        splits=["final", "tuning"],
        out_dir=tmp_path / "answer-quality",
        resamples=0,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[2] for entry in seen] == [("q1",), ("q2",), ("q1",), ("q2",)]
    assert run.report["item_ids"] == ["q1", "q2"]
    assert run.report["verdict"]["focus_n"] == 2
    assert len(run.report["lanes"][VECTOR]["run_dirs"]) == 2


def test_a_split_that_selects_nothing_fails_instead_of_shrinking_the_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(SystemExit, match="tuning"):
        run_answer_quality(
            RunConfig(data_dir=tmp_path, goldset_path=goldset),
            parse_lanes(f"{VECTOR},{FUSED}"),
            splits=["final", "tuning"],
            run_lane=_recording_lane(tmp_path, []),
        )
