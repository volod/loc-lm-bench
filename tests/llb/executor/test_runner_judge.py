"""Tests for runner judge."""

import pytest
from llb.backends.base import ChatResult
from llb.core.config import RunConfig
from llb.executor.runner import run_eval
from test_runner import DOC, FakeLauncher, FakeStore, _runner_fn, gold_item


def test_run_eval_wires_trusted_judge_and_persists_scores(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    store = FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )
    launcher = FakeLauncher(lambda _m: ChatResult(text="Київ", completion_tokens=2, latency_s=0.4))
    cfg = RunConfig(
        data_dir=tmp_path,
        run_name="judge-test",
        top_k=3,
        model="fake-uk",
        judge_model="judge-x",
        judge_base_url="http://localhost:9000/v1",
    )
    calls = {}

    def scorer(records, judge_model):
        calls["model"] = judge_model
        assert records[0]["question"] == q and records[0]["answer"] == "Київ"
        assert records[0]["contexts"]  # retrieved chunk text passed through
        return [{"faithfulness": 1.0, "answer_relevancy": 0.8} for _ in records]

    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        judge_rho=0.7,  # >= 0.6 threshold -> judge trusted -> enters the blend
        judge_scorer=scorer,
        emit=False,
    )

    assert calls["model"] == "judge-x"
    assert result["metrics"]["judge_score"] == 0.9  # mean(1.0, 0.8)
    row = result["rows"][0]
    assert row["judge"] == 0.9
    assert row["quality"] == 0.95  # blend objective 1.0 with judge 0.9 at weight 0.5
    assert result["manifest"].judge["provider"] == "deepeval-geval"
    assert result["manifest"].judge["base_url"] == "http://localhost:9000/v1"


def test_run_eval_demotes_uncalibrated_judge(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    store = FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )
    launcher = FakeLauncher(lambda _m: ChatResult(text="Київ", completion_tokens=2, latency_s=0.4))
    cfg = RunConfig(data_dir=tmp_path, run_name="demote", model="fake-uk", judge_model="judge-x")

    def scorer(records, judge_model):  # pragma: no cover - must NOT run when demoted
        raise AssertionError("judge must not run without calibration")

    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        judge_rho=None,  # uncalibrated -> demoted -> objective ranks alone
        judge_scorer=scorer,
        emit=False,
    )
    assert "judge_score" not in result["metrics"]
    assert result["rows"][0]["judge"] is None


def test_run_eval_errors_when_split_empty(tmp_path):
    cfg = RunConfig(data_dir=tmp_path, run_name="empty")
    try:
        run_eval(
            cfg,
            items=[],
            launcher=FakeLauncher(lambda m: ChatResult(text="")),
            runner_fn=lambda it: {},
            emit=False,
        )
        raise AssertionError("expected SystemExit on empty eval set")
    except SystemExit:
        pass


@pytest.mark.slow
def test_run_eval_emits_prefilled_worksheet(tmp_path):
    items = [gold_item("cal-1", "Яка столиця України?", "Київ", "Київ", split="calibration")]
    store = FakeStore(
        {
            "Яка столиця України?": [
                {"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}
            ]
        }
    )
    launcher = FakeLauncher(
        lambda messages: ChatResult(text="Київ", completion_tokens=2, latency_s=0.3)
    )
    cfg = RunConfig(data_dir=tmp_path, run_name="cal", model="fake-uk")
    ws = tmp_path / "worksheet.csv"

    run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        split="calibration",
        worksheet=ws,
        emit=False,
    )

    text = ws.read_text(encoding="utf-8")
    assert "cal-1" in text and "Київ" in text and "human_rating" in text


def test_worksheet_prefills_judge_rating_ungated(tmp_path):
    import csv

    q = "Яка столиця України?"
    items = [gold_item("cal-1", q, "Київ", "Київ", split="calibration")]
    store = FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )
    launcher = FakeLauncher(
        lambda messages: ChatResult(text="Київ", completion_tokens=2, latency_s=0.3)
    )
    cfg = RunConfig(data_dir=tmp_path, run_name="cal-judge", model="fake-uk", judge_model="judge-x")
    ws = tmp_path / "worksheet.csv"

    def scorer(records, judge_model):
        return [{"faithfulness": 1.0, "answer_relevancy": 0.6} for _ in records]

    run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        split="calibration",
        worksheet=ws,
        judge_rho=None,  # ungated for the worksheet: judge runs even when not (yet) calibrated
        judge_scorer=scorer,
        emit=False,
    )

    rows = list(csv.DictReader(ws.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["judge_rating"] == "0.8"  # mean(1.0, 0.6), pre-filled by the ungated judge
    assert rows[0]["human_rating"] == ""  # human still fills this column
