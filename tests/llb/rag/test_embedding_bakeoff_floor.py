"""Focused tests split from ``test_embedding_bakeoff.py``."""

from _embedding_bakeoff_helpers import (
    _RESOLVED_SECOND_ITEM,
    _TIED_ITEM_REORDERED,
    _chunk,
    _FakeStore,
    _fixed_builder,
    _floor_bakeoff,
    _items,
)

from llb.rag.embedding_bakeoff import run_bakeoff
from llb.rag.embedding_bakeoff_report import (
    format_report,
    render_markdown,
)


def test_bakeoff_floor_is_opt_in():
    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["e5"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 0, 10)])),
    )
    assert "noise_floor" not in report  # default bake-off row set is unchanged


def test_bakeoff_floor_covers_every_candidate_lane():
    report = _floor_bakeoff(_TIED_ITEM_REORDERED)
    floor = report["noise_floor"]
    assert set(floor["lanes"]) == {"cand-a", "cand-b"}
    assert floor["floor_recall_at_k"] > 0.0  # the tied item can land either side of the cut
    # Each lane's floor base is the recall the candidate row published: the two cannot drift.
    by_model = {row["model"]: row for row in report["candidates"]}
    for model, lane in floor["lanes"].items():
        assert lane["recall_at_k"]["base"] == by_model[model]["recall_at_k"]


def test_bakeoff_recommendation_that_rests_on_tie_order_does_not_clear_the_floor():
    report = _floor_bakeoff(_TIED_ITEM_REORDERED)
    margin = report["noise_floor"]["margin"]
    assert margin["leader"] == report["best_recall"]
    assert margin["clears_floor"] is False
    md = render_markdown(report)
    assert "Measurement floor" in md and "does NOT clear the floor" in md
    assert md.isascii()  # AGENTS.md: ASCII-only output
    assert "noise floor" in format_report(report)


def test_bakeoff_recommendation_on_a_resolved_lead_clears_the_floor():
    report = _floor_bakeoff(_RESOLVED_SECOND_ITEM)
    margin = report["noise_floor"]["margin"]
    assert margin["leader"] == "cand-b" == report["best_recall"]
    assert margin["delta"] > margin["floor"] and margin["clears_floor"] is True
    assert "clears the floor" in render_markdown(report)


def test_markdown_says_the_floor_is_unmeasured_when_it_was_not_asked_for():
    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["e5"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 0, 10)])),
    )
    assert "--noise-floor" in render_markdown(report)
