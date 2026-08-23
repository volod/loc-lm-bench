"""Focused tests split from ``test_context_ablation.py``."""

import json
from pathlib import Path

import pytest
from tests.llb.eval._context_ablation_helpers import (
    _derived,
    _gold_item,
    _recording_lane,
    _write_bundle,
)

from llb.core.config import RunConfig
from llb.eval.context_ablation import run_context_ablation
from llb.eval.context_ablation.models import (
    DERIVED_RETRIEVAL_UPLIFT,
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    LANE_RETRIEVED_DOCUMENT,
)


def test_every_lane_scores_the_same_selected_items_and_the_comparison_persists(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        out_dir=tmp_path / "context-ablation",
        resamples=50,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[1] for entry in seen] == [
        LANE_CLOSED_BOOK,
        LANE_RAG,
        LANE_RETRIEVED_DOCUMENT,
        LANE_LONG_CONTEXT,
    ]
    assert {entry[2] for entry in seen} == {("q1", "q2")}
    assert run.report["item_ids"] == ["q1", "q2"]
    assert run.report["lanes"][LANE_RAG]["run_dirs"] == [
        str(tmp_path / "run-eval" / f"context-ablation-{LANE_RAG}-final")
    ]
    assert _derived(run.report, DERIVED_RETRIEVAL_UPLIFT)["paired"]["delta"][
        "mean"
    ] == pytest.approx(0.5)
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["split"] == "final"
    assert persisted["metadata"]["grounding"] == "verified"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# RAG versus")


def test_the_baseline_lane_cannot_be_dropped_from_the_selection(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(ValueError, match="closed_book"):
        run_context_ablation(
            RunConfig(data_dir=tmp_path, goldset_path=goldset), [LANE_RAG, LANE_LONG_CONTEXT]
        )


def test_a_single_lane_is_not_a_comparison(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(ValueError, match="at least one other lane"):
        run_context_ablation(RunConfig(data_dir=tmp_path, goldset_path=goldset), [LANE_CLOSED_BOOK])


def test_several_splits_pool_into_one_compared_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    items = [_gold_item("q1"), _gold_item("q2")]
    items[1].split = "tuning"
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8"
    )
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_CLOSED_BOOK, LANE_RAG],
        splits=["final", "tuning"],
        out_dir=tmp_path / "context-ablation",
        resamples=0,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[2] for entry in seen] == [("q1",), ("q2",), ("q1",), ("q2",)]
    assert run.report["item_ids"] == ["q1", "q2"]
    assert len(run.report["lanes"][LANE_RAG]["run_dirs"]) == 2


def test_a_split_that_selects_nothing_fails_instead_of_shrinking_the_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(SystemExit, match="tuning"):
        run_context_ablation(
            RunConfig(data_dir=tmp_path, goldset_path=goldset),
            splits=["final", "tuning"],
            run_lane=_recording_lane(tmp_path, []),
        )
