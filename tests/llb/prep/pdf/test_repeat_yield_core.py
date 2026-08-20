"""Per-question yield audit for `--repeat-blocks drop` (`llb.prep.pdf.repeat_yield`).

Pure: fake stores exposing the `.retrieve` seam and hand-built gold items, plus one end-to-end
CLI test over the committed intra-document-repeats fixture with a straddling item. No FAISS, no
embedder, no GPU on the pure lane.
"""

from llb.prep.pdf.repeat_yield import (
    VERDICT_DROPPED,
    VERDICT_HELD,
    VERDICT_LOST,
    VERDICT_RECOVERED,
    audit_repeat_yield,
    format_yield_report,
)


from tests.llb.prep._repeat_yield_helpers import (
    _StubStore,
    _item,
    _chunk,
)


def test_held_recovered_lost_and_dropped_verdicts():
    baseline = [_item("held", 0, 5), _item("lost", 10, 15), _item("gone", 20, 25)]
    # the stripped set drops "gone" (straddled) and re-homes "held"+"lost" onto a survivor at 100
    stripped = [_item("held", 100, 105), _item("lost", 100, 105)]
    # baseline could answer all three (incl. the to-be-dropped "gone")
    baseline_store = _StubStore(
        {"held": _chunk(0, 5), "lost": _chunk(10, 15), "gone": _chunk(20, 25)}
    )
    stripped_store = _StubStore({"held": _chunk(100, 105)})  # "lost" no longer retrieved

    report = audit_repeat_yield(
        baseline,
        stripped,
        baseline_store,
        stripped_store,
        dropped_ids={"gone"},
        rehomed_ids={"held", "lost"},
        k=10,
    )

    verdicts = {entry["id"]: entry["verdict"] for entry in report["moved"]}
    assert verdicts == {"held": VERDICT_HELD, "lost": VERDICT_LOST, "gone": VERDICT_DROPPED}
    assert report["lost"] == ["lost", "gone"]  # lost re-home + dropped item the baseline could hit
    assert report["adopt"] is False
    assert report["baseline_recall"] == 1.0  # all three baseline items hit
    assert report["stripped_recall"] == 0.5  # 1 of the 2 scored (non-dropped) items hit


def test_adopt_when_every_touched_question_is_held_or_recovered():
    baseline = [_item("held", 0, 5), _item("recovered", 10, 15)]
    stripped = [_item("held", 100, 105), _item("recovered", 100, 105)]
    baseline_store = _StubStore({"held": _chunk(0, 5)})  # baseline misses "recovered"
    stripped_store = _StubStore({"held": _chunk(100, 105), "recovered": _chunk(100, 105)})

    report = audit_repeat_yield(
        baseline,
        stripped,
        baseline_store,
        stripped_store,
        dropped_ids=set(),
        rehomed_ids={"held", "recovered"},
        k=10,
    )

    verdicts = {entry["id"]: entry["verdict"] for entry in report["moved"]}
    assert verdicts == {"held": VERDICT_HELD, "recovered": VERDICT_RECOVERED}
    assert report["adopt"] is True
    assert report["lost"] == []


def test_dropped_item_the_baseline_missed_is_not_counted_as_lost():
    baseline = [_item("gone", 0, 5)]
    baseline_store = _StubStore({})  # baseline could not answer it anyway
    stripped_store = _StubStore({})

    report = audit_repeat_yield(
        baseline, [], baseline_store, stripped_store, dropped_ids={"gone"}, rehomed_ids=set(), k=10
    )

    assert report["lost"] == []
    assert report["adopt"] is True
    assert report["moved"][0]["verdict"] == VERDICT_DROPPED


def test_format_is_ascii_and_states_the_verdict():
    report = audit_repeat_yield(
        [_item("lost", 0, 5)],
        [_item("lost", 100, 105)],
        _StubStore({"lost": _chunk(0, 5)}),
        _StubStore({}),
        dropped_ids=set(),
        rehomed_ids={"lost"},
        k=10,
    )
    text = format_yield_report(report)
    assert text.isascii()
    assert "HOLD" in text and "lost" in text
    assert "pooled recall@10" in text
