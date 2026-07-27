"""Focused tests split from ``test_embedder_adoption_roster.py``."""

import json
from pathlib import Path

import pytest
from _embedder_adoption_roster_helpers import (
    FOCUS,
    _report,
    _roster,
)

from llb.eval.embedder_adoption.roster_models import (
    DECISION_NO_PROPERTY_PREDICTS,
    DECISION_PROPERTY_PREDICTS,
)
from llb.eval.embedder_adoption.roster import compare_roster
from llb.eval.embedder_adoption.roster_report import (
    format_roster,
    format_roster_summary,
)
from llb.eval.embedder_adoption.comparison_run import (
    load_profiles,
    run_roster_comparison,
)


def test_profiles_are_declared_and_validated(tmp_path: Path):
    good = tmp_path / "profiles.json"
    good.write_text(json.dumps({"a": {"params_b": 27, "family": "gemma"}}), encoding="utf-8")
    assert load_profiles(good) == {"a": {"params_b": 27, "family": "gemma"}}
    assert load_profiles(None) == {}

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": {"parameters": 27}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown propert"):
        load_profiles(bad)

    not_object = tmp_path / "list.json"
    not_object.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="keyed by model id"):
        load_profiles(not_object)


def test_roster_report_renders_ascii_with_readings_properties_and_separations():
    roster = _roster(
        [("big-a", True), ("big-b", True), ("small-a", False), ("small-b", False)],
        {
            "big-a": {"params_b": 27, "family": "gemma"},
            "big-b": {"params_b": 24, "family": "gemma"},
            "small-a": {"params_b": 12, "family": "qwen"},
            "small-b": {"params_b": 8, "family": "qwen"},
        },
    )
    text = format_roster(roster, metadata={"goldset": "gs.jsonl"})
    assert "# Embedder adoption bar" in text
    assert "### Per-cell reading by model" in text
    assert "### Declared model properties" in text
    assert "### Property separation" in text
    assert DECISION_PROPERTY_PREDICTS in text
    assert text.isascii()
    assert format_roster_summary(roster).isascii()


def test_roster_round_trips_through_disk(tmp_path: Path):
    dirs = []
    for name, answers in (("a", True), ("b", False), ("c", False)):
        d = tmp_path / name
        d.mkdir()
        (d / "comparison.json").write_text(
            json.dumps(_report(model=name, focus_answers=answers), ensure_ascii=False),
            encoding="utf-8",
        )
        dirs.append(d)
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps({"a": {"params_b": 27}, "b": {"params_b": 12}, "c": {"params_b": 8}}),
        encoding="utf-8",
    )

    run = run_roster_comparison(
        dirs, out_dir=tmp_path / "roster", profiles_path=profiles, focus_cell=FOCUS
    )
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["verdict"]["decision"] == DECISION_PROPERTY_PREDICTS
    assert persisted["metadata"]["goldset"] == "gs.jsonl"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Embedder adoption")


def test_the_roster_reads_the_stability_each_sweep_persisted_without_touching_a_bundle():
    """A sweep measures its own cells, so a roster over it needs no run bundles at all."""
    roster = compare_roster(
        [_report(model=m, focus_answers=a) for m, a in (("a", True), ("b", False), ("c", False))],
        {},
        focus_cell=FOCUS,
        measure_stability=True,  # the fake reports name no real bundles
    )
    focus = next(cell for cell in roster["cells"] if cell["label"] == FOCUS)
    assert set(focus["stability"]) == {"a", "b", "c"}
    assert roster["verdict"]["decision"] == DECISION_NO_PROPERTY_PREDICTS


def test_stability_is_absent_for_a_sweep_that_carries_none_and_whose_bundles_are_gone():
    """An archived roster still reads: the annotation is additive, never load-bearing."""
    reports = [
        _report(model=m, focus_answers=a) for m, a in (("a", True), ("b", False), ("c", False))
    ]
    for report in reports:  # a sweep recorded before the annotation existed
        for cell in report["cells"]:
            cell.pop("stability", None)
    roster = compare_roster(reports, {}, focus_cell=FOCUS, measure_stability=True)
    assert all("stability" not in cell for cell in roster["cells"])
    assert roster["verdict"]["decision"] == DECISION_NO_PROPERTY_PREDICTS
    assert format_roster(roster).count("borderline") >= 0  # renders without a stability map


def test_measuring_stability_never_changes_a_reading_or_the_verdict():
    spec = [("a", True), ("b", False), ("c", False)]
    off = compare_roster(
        [_report(model=m, focus_answers=x) for m, x in spec],
        {},
        focus_cell=FOCUS,
        measure_stability=False,
    )
    on = compare_roster(
        [_report(model=m, focus_answers=x) for m, x in spec],
        {},
        focus_cell=FOCUS,
        measure_stability=True,
    )
    assert [c["readings"] for c in off["cells"]] == [c["readings"] for c in on["cells"]]
    assert off["verdict"]["decision"] == on["verdict"]["decision"]
