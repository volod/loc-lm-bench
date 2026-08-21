"""Focused tests split from ``test_query_robustness.py``."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.llb.eval._query_robustness_helpers import (
    APOSTROPHE_QUESTION,
    FakeEndpoint,
    FakeStore,
    _item,
    build_fake_graph,
)

from llb.core.config import RunConfig, restoration_fields
from llb.eval.query_robustness.evaluate import (
    LANE_NORMALIZE,
    LANE_NORMALIZE_TYPOS,
    LANE_OFF,
    MITIGATION_LANES,
    LaneMetrics,
    RobustnessResult,
    evaluate_query_robustness,
    mitigation_lanes_for_class,
)
from llb.eval.query_robustness.report import write_robustness_artifacts
from llb.eval.query_robustness.run import make_query_executor
from llb.eval.query_robustness.variants import (
    APOSTROPHE_MIXED_SCRIPT,
    APOSTROPHE_VARIANT,
    KEYBOARD_TYPOS,
    MIXED_SCRIPT,
    TRANSLITERATION,
    VARIANT_CLASSES,
)
from llb.goldset.schema import GoldItem


@dataclass
class _GuardLoader:
    calls: list[bool]

    def __call__(self) -> Callable[[str], bool]:
        self.calls.append(True)
        return lambda _token: False


def _lane_index(result: RobustnessResult) -> dict[tuple[str, str], LaneMetrics]:
    return {(lane.variant_class, lane.mitigation): lane for lane in result.lanes}


def _assert_probe_matrix(result: RobustnessResult, guard_loaded: list[bool]) -> None:
    assert len(result.rows) == len(VARIANT_CLASSES) * len(MITIGATION_LANES)
    assert guard_loaded == [True]
    assert all(row["probe"] is True for row in result.rows)
    assert {lane.mitigation for lane in result.lanes} == {lane.id for lane in MITIGATION_LANES}


def _assert_mitigation_recovery(lanes: dict[tuple[str, str], LaneMetrics]) -> None:
    for variant_class in (TRANSLITERATION, MIXED_SCRIPT, KEYBOARD_TYPOS):
        assert lanes[(variant_class, LANE_OFF.id)].recall_at_k == 0.0
        assert lanes[(variant_class, LANE_NORMALIZE_TYPOS.id)].recall_at_k == 1.0
        assert lanes[(variant_class, LANE_NORMALIZE_TYPOS.id)].recall_recovery == 1.0

    assert lanes[(TRANSLITERATION, LANE_NORMALIZE.id)].recall_at_k == 1.0
    assert lanes[(MIXED_SCRIPT, LANE_NORMALIZE.id)].recall_at_k == 1.0
    assert lanes[(KEYBOARD_TYPOS, LANE_NORMALIZE.id)].recall_at_k == 0.0
    assert APOSTROPHE_MIXED_SCRIPT not in {lane.variant_class for lane in lanes.values()}
    assert lanes[(APOSTROPHE_VARIANT, LANE_OFF.id)].recall_at_k == 1.0
    assert lanes[(APOSTROPHE_VARIANT, LANE_OFF.id)].changed.n == 0


def _write_probe_artifacts(result: RobustnessResult, out: Path) -> str:
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
    return (out / "report.md").read_text(encoding="utf-8")


def _assert_report_content(report: str) -> None:
    # Lane ids are identical with and without dense-lane casing, so the header states which run
    # this report is -- otherwise the two are indistinguishable side by side.
    assert "- dense-lane casing: off" in report
    assert f"| {APOSTROPHE_VARIANT} | `{LANE_NORMALIZE.id}` |" in report
    assert f"| {MIXED_SCRIPT} | `{LANE_NORMALIZE.id}` |" in report
    assert "## Paired uncertainty by noise class" in report
    assert "p_positive" in report
    assert f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | changed |" not in report
    assert (
        f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | 0 | - | - | - | - | - | - | - | - | - |"
        in report
    )


def _assert_uncertainty_payload(lanes: dict[tuple[str, str], LaneMetrics]) -> None:
    transliteration = lanes[(TRANSLITERATION, LANE_OFF.id)]
    assert transliteration.comparisons["recall_delta"]["delta"]["mean"] == (
        transliteration.recall_delta
    )
    assert "stability" in transliteration.comparisons["recall_delta"]


def test_fake_store_endpoint_measure_mitigation_and_keep_probe_rows_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    item = _item()
    guard_loaded: list[bool] = []

    monkeypatch.setattr(
        "llb.rag.vector_store.lexical.load_uk_word_probe", _GuardLoader(guard_loaded)
    )
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    executor = make_query_executor(RunConfig(top_k=1, max_tokens=16), FakeStore(), FakeEndpoint())
    clean_rows = [{"item_id": item.id, "objective_score": 1.0, "retrieval_hit": 1.0}]
    result = evaluate_query_robustness([item], clean_rows, executor, seed=13, typo_rate=0.1)

    _assert_probe_matrix(result, guard_loaded)
    lanes = _lane_index(result)
    _assert_mitigation_recovery(lanes)
    out = tmp_path / "query-robustness" / "run"
    report = _write_probe_artifacts(result, out)
    _assert_report_content(report)
    _assert_uncertainty_payload(lanes)


def test_items_a_class_cannot_perturb_are_reported_apart_from_the_affected_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "llb.rag.vector_store.lexical.load_uk_word_probe", lambda: lambda _token: False
    )
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
        f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | 1 | 0.0000 | -1.0000 | 0.0000 | "
        "-1.0000 | 0.0000 | -1.0000 |"
    ) in report
    assert (
        f"| {APOSTROPHE_VARIANT} | `{LANE_NORMALIZE.id}` | 1 | 1.0000 | +0.0000 | 1.0000 "
        "| +0.0000 | 1.0000 | +0.0000 | +1.0000 | +1.0000 | +1.0000 |"
    ) in report


class CaseSensitiveStore(FakeStore):
    """A dense store that only matches the question's ORIGINAL capitalization.

    The real e5 encoder is case-sensitive by degrees; this fake makes the same property exact so
    the dense-case lane's effect is a hit or a miss rather than a score wobble.
    """

    def __init__(self, *questions: str) -> None:
        super().__init__(*questions)
        self.questions = set(questions or (_item().question,))

    def retrieve(self, question: str, k: int) -> list[dict[str, object]]:
        return [self.chunk] if question in self.questions else []


def test_dense_case_lane_flags_only_the_mitigated_lanes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "llb.rag.vector_store.lexical.load_uk_word_probe", lambda: lambda _token: False
    )
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    item = _item()
    store = CaseSensitiveStore(item.question)
    lanes = mitigation_lanes_for_class(APOSTROPHE_VARIANT, dense_case=True)
    executor = make_query_executor(
        RunConfig(top_k=1, max_tokens=16), store, FakeEndpoint(), lanes=lanes
    )
    clean_rows = [{"item_id": item.id, "objective_score": 1.0, "retrieval_hit": 1.0}]
    result = evaluate_query_robustness(
        [item],
        clean_rows,
        executor,
        seed=13,
        typo_rate=0.1,
        variant_classes=[APOSTROPHE_VARIANT],
        dense_case=True,
    )

    flags = {row["mitigation"]: row["mitigation_dense_case"] for row in result.rows}
    assert flags == {LANE_OFF.id: False, LANE_NORMALIZE.id: True, LANE_NORMALIZE_TYPOS.id: True}
    # The lane ids are unchanged, so a dense-case report reads cell for cell against a folded one.
    assert [lane.id for lane in lanes] == [lane.id for lane in MITIGATION_LANES]


def test_dense_case_lane_recovers_the_hit_casefolding_costs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "llb.rag.vector_store.lexical.load_uk_word_probe", lambda: lambda _token: False
    )
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    item = _item()
    clean_rows = [{"item_id": item.id, "objective_score": 1.0, "retrieval_hit": 1.0}]

    def recall(dense_case: bool) -> float:
        lanes = mitigation_lanes_for_class(APOSTROPHE_VARIANT, dense_case=dense_case)
        executor = make_query_executor(
            RunConfig(top_k=1, max_tokens=16),
            CaseSensitiveStore(item.question),
            FakeEndpoint(),
            lanes=lanes,
        )
        result = evaluate_query_robustness(
            [item],
            clean_rows,
            executor,
            seed=13,
            typo_rate=0.1,
            variant_classes=[APOSTROPHE_VARIANT],
            dense_case=dense_case,
        )
        return {lane.mitigation: lane for lane in result.lanes}[LANE_NORMALIZE.id].recall_at_k

    # The class perturbs nothing in this question, so the fold is the ONLY thing the lane changes.
    assert recall(False) == 0.0
    assert recall(True) == 1.0


def test_a_swept_restoration_setting_reaches_the_typo_lane_and_no_other(
    monkeypatch: pytest.MonkeyPatch,
):
    """A non-default restoration constant is refused in a lane without the step it constrains."""
    from llb.eval.query_robustness.run import _baseline_config
    from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY

    built: list[tuple[tuple[str, ...], int]] = []

    def _record(config, store, launcher):
        built.append((tuple(config.query_prep), config.query_prep_ambiguous_max_chars))
        return None

    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    monkeypatch.setattr("llb.eval.query_robustness.run.build_query_prep", _record)
    config = RunConfig(
        top_k=1,
        max_tokens=16,
        query_prep=["normalize", "typos"],
        query_prep_ambiguous_max_chars=3,
    )
    # the clean baseline drops the whole lane, so it may not carry the constant either
    assert _baseline_config(config).query_prep_ambiguous_max_chars == (
        DEFAULT_RESTORATION_POLICY.ambiguous_token_max_chars
    )
    make_query_executor(
        _baseline_config(config),
        FakeStore(),
        FakeEndpoint(),
        restoration=restoration_fields(config),
    )
    assert (("normalize", "typos"), 3) in built
    assert (("normalize",), DEFAULT_RESTORATION_POLICY.ambiguous_token_max_chars) in built
