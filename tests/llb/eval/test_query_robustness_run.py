"""Focused tests split from ``test_query_robustness.py``."""

from pathlib import Path

import pytest
from _query_robustness_helpers import (
    APOSTROPHE_QUESTION,
    FakeEndpoint,
    FakeStore,
    _item,
    build_fake_graph,
)

from llb.core.config import RunConfig
from llb.eval.query_robustness import (
    LANE_NORMALIZE,
    LANE_NORMALIZE_TYPOS,
    LANE_OFF,
    MITIGATION_LANES,
    evaluate_query_robustness,
)
from llb.eval.query_robustness_report import write_robustness_artifacts
from llb.eval.query_robustness_run import make_query_executor
from llb.eval.query_robustness_variants import (
    APOSTROPHE_MIXED_SCRIPT,
    APOSTROPHE_VARIANT,
    KEYBOARD_TYPOS,
    MIXED_SCRIPT,
    TRANSLITERATION,
    VARIANT_CLASSES,
)
from llb.goldset.schema import GoldItem


def test_fake_store_endpoint_measure_mitigation_and_keep_probe_rows_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    item = _item()
    guard_loaded = []

    def load_guard():
        guard_loaded.append(True)
        return lambda _token: False

    monkeypatch.setattr("llb.rag.lexical.load_uk_word_probe", load_guard)
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    executor = make_query_executor(RunConfig(top_k=1, max_tokens=16), FakeStore(), FakeEndpoint())
    clean_rows = [{"item_id": item.id, "objective_score": 1.0, "retrieval_hit": 1.0}]
    result = evaluate_query_robustness([item], clean_rows, executor, seed=13, typo_rate=0.1)

    assert len(result.rows) == len(VARIANT_CLASSES) * len(MITIGATION_LANES)
    assert guard_loaded == [True]  # only the vocabulary-correction lane loads the morphology probe
    assert all(row["probe"] is True for row in result.rows)
    lanes = {(lane.variant_class, lane.mitigation): lane for lane in result.lanes}
    assert {lane.mitigation for lane in result.lanes} == {lane.id for lane in MITIGATION_LANES}
    for variant_class in (TRANSLITERATION, MIXED_SCRIPT, KEYBOARD_TYPOS):
        assert lanes[(variant_class, LANE_OFF.id)].recall_at_k == 0.0
        assert lanes[(variant_class, LANE_NORMALIZE_TYPOS.id)].recall_at_k == 1.0
        assert lanes[(variant_class, LANE_NORMALIZE_TYPOS.id)].recall_recovery == 1.0
    # the isolated lanes separate the two mechanisms: normalization alone inverts the noise it can
    # attribute (script and transliteration), and only keyboard typos need vocabulary correction
    assert lanes[(TRANSLITERATION, LANE_NORMALIZE.id)].recall_at_k == 1.0
    assert lanes[(MIXED_SCRIPT, LANE_NORMALIZE.id)].recall_at_k == 1.0
    assert lanes[(KEYBOARD_TYPOS, LANE_NORMALIZE.id)].recall_at_k == 0.0
    # each mechanism is its own class, so a recovery is attributable to one of them; the combined
    # class is opt-in, and a class with nothing to perturb reproduces the clean rows exactly
    assert APOSTROPHE_MIXED_SCRIPT not in {lane.variant_class for lane in result.lanes}
    assert lanes[(APOSTROPHE_VARIANT, LANE_OFF.id)].recall_at_k == 1.0
    assert lanes[(APOSTROPHE_VARIANT, LANE_OFF.id)].changed.n == 0

    out = tmp_path / "query-robustness" / "run"
    paths = write_robustness_artifacts(
        result,
        out,
        {
            "model": "fake",
            "backend": "fake",
            "split": "final",
            "seed": 13,
            "typo_rate": 0.1,
            "clean_run_dir": "run-eval/clean",
        },
    )
    assert set(out.iterdir()) == {Path(paths["report"]), Path(paths["robustness"])}
    assert not (out / "scores.jsonl").exists()
    assert len((out / "robustness.jsonl").read_text(encoding="utf-8").splitlines()) == len(
        result.rows
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    assert f"| {APOSTROPHE_VARIANT} | `{LANE_NORMALIZE.id}` |" in report
    assert f"| {MIXED_SCRIPT} | `{LANE_NORMALIZE.id}` |" in report
    assert "## Paired uncertainty by noise class" in report
    assert "p_positive" in report
    assert f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | changed |" not in report
    transliteration = lanes[(TRANSLITERATION, LANE_OFF.id)]
    assert transliteration.comparisons["recall_delta"]["delta"]["mean"] == (
        transliteration.recall_delta
    )
    assert "stability" in transliteration.comparisons["recall_delta"]
    # the untouched apostrophe rows carry no measurement, so the subset table dashes them out
    assert f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | 0 | - | - | - | - | - | - |" in report


def test_items_a_class_cannot_perturb_are_reported_apart_from_the_affected_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("llb.rag.lexical.load_uk_word_probe", lambda: lambda _token: False)
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    plain_item = _item()
    apostrophe_item = GoldItem(
        **{**plain_item.model_dump(), "id": "q2", "question": APOSTROPHE_QUESTION}
    )
    store = FakeStore(apostrophe_item.question, plain_item.question)
    executor = make_query_executor(RunConfig(top_k=1, max_tokens=16), store, FakeEndpoint())
    clean_rows = [
        {"item_id": item.id, "objective_score": 1.0, "retrieval_hit": 1.0}
        for item in (apostrophe_item, plain_item)
    ]
    result = evaluate_query_robustness(
        [apostrophe_item, plain_item],
        clean_rows,
        executor,
        seed=13,
        typo_rate=0.1,
        variant_classes=[APOSTROPHE_VARIANT],
    )

    assert result.variant_classes == (APOSTROPHE_VARIANT,)
    lanes = {lane.mitigation: lane for lane in result.lanes}
    off, normalized = lanes[LANE_OFF.id], lanes[LANE_NORMALIZE.id]
    assert (off.n, off.changed.n) == (2, 1)
    # pooling the item the class cannot touch halves the visible loss; the subset states it whole
    assert (off.recall_at_k, off.recall_delta) == (0.5, -0.5)
    assert (off.changed.recall_at_k, off.changed.recall_delta) == (0.0, -1.0)
    assert normalized.recall_recovery == 0.5
    assert normalized.changed.recall_recovery == 1.0

    out = tmp_path / "query-robustness" / "run"
    write_robustness_artifacts(
        result,
        out,
        {
            "model": "fake",
            "backend": "fake",
            "split": "final",
            "seed": 13,
            "typo_rate": 0.1,
            "clean_run_dir": "run-eval/clean",
        },
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Affected items only" in report
    assert f"`{APOSTROPHE_VARIANT}` 1" in report
    assert (
        f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | 1 | 0.0000 | -1.0000 | 0.0000 | -1.0000 |"
    ) in report
    assert (
        f"| {APOSTROPHE_VARIANT} | `{LANE_NORMALIZE.id}` | 1 | 1.0000 | +0.0000 | 1.0000 "
        "| +0.0000 | +1.0000 | +1.0000 |"
    ) in report
