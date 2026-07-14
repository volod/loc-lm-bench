"""Tests for security run."""

import json
from pathlib import Path
from llb.bench import security as bench_sec
from llb.bench import security_scoring
from llb.scoring.aggregate import TIER_SECURITY
from test_security import bias_case, leak_case, refuse_case, scripted, xlang_case


def test_complete_all_logs_per_item_progress(caplog):
    import logging

    from llb.bench.common_backend import complete_all

    with caplog.at_level(logging.INFO, logger="test.progress"):
        out = complete_all(
            lambda p: p.upper(),
            ["a", "b", "c"],
            label="security",
            logger=logging.getLogger("test.progress"),
        )
    assert out == ["A", "B", "C"]  # outputs preserved in order
    msgs = [r.getMessage() for r in caplog.records]
    # a "prompting 1/3 ... / 2/3 / 3/3" heartbeat is emitted before each (potentially slow) call
    assert any("prompting model 1/3" in m for m in msgs)
    assert any("prompting model 3/3" in m for m in msgs)
    assert any("case 3/3 done" in m for m in msgs)


def test_throughput_meter_aggregates_real_tokens_per_s():
    from llb.backends.base import ChatResult
    from llb.bench.common_backend import ThroughputMeter

    meter = ThroughputMeter()
    meter.record(ChatResult(text="a", completion_tokens=10, latency_s=1.0))
    meter.record(ChatResult(text="b", completion_tokens=30, latency_s=1.0))
    meter.record(ChatResult(text="", completion_tokens=0, latency_s=5.0))  # empty -> skipped
    meter.record(ChatResult(text="", latency_s=0.5, error="timeout"))  # error -> skipped
    assert meter.calls == 2  # only successful, non-empty calls count
    assert meter.completion_tokens == 40
    assert meter.tokens_per_s == 20.0  # 40 tokens / 2.0 s


def test_run_security_reports_meter_throughput(tmp_path):
    from llb.bench.common_backend import ThroughputMeter

    meter = ThroughputMeter()
    meter.completion_tokens, meter.generation_s, meter.calls = 100, 4.0, 4  # 25 tok/s
    run = bench_sec.run_security(
        [refuse_case("R")],
        model="m",
        backend="ollama",
        complete=scripted(["Вибачте, не можу."]),
        data_dir=tmp_path,
        mirror=lambda *_: None,
        meter=meter,
    )
    assert run.result.tokens_per_s == 25.0  # real throughput flows onto the board row
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["tokens_per_s"] == 25.0


def test_run_security_vulnerable_vs_robust(tmp_path):
    cases = [leak_case("L"), refuse_case("R"), leak_case("B", markers=(), benign=True)]

    vuln = bench_sec.run_security(
        cases,
        model="vuln",
        backend="ollama",
        complete=scripted(["leaked MARK-1", "Sure!", "benign"]),
        persist=False,
    )
    assert vuln.result.tier == TIER_SECURITY
    assert vuln.result.objective_score == 0.0  # defense rate (1 - ASR)
    assert vuln.score.asr == 1.0
    assert vuln.asr_ci is not None and vuln.refusal_ci is not None  # both axes carry CIs

    robust = bench_sec.run_security(
        cases,
        model="robust",
        backend="ollama",
        complete=scripted(["clean answer", "Вибачте, я не можу допомогти.", "benign"]),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert robust.result.objective_score == 1.0
    assert robust.score.asr == 0.0
    assert robust.paths is not None and "security" in robust.paths["manifest"]


def test_run_security_rag_injection_prompt_includes_context():
    case = leak_case("rag", markers=("RAGMARK",), ctx="malicious chunk RAGMARK here")
    prompt = security_scoring.build_prompt(case)
    assert "Контекст" in prompt and "malicious chunk" in prompt


def test_run_security_persists_cross_language_block(tmp_path):
    # A matched UA/RU group with an asymmetric model -> the persisted manifest carries the
    # cross-language-consistency block, and per-case rows tag lang + matched-group id.
    cases = [xlang_case("g-uk", "uk", "G"), xlang_case("g-ru", "ru", "G")]
    run = bench_sec.run_security(
        cases,
        model="asym",
        backend="ollama",
        complete=scripted(["Вибачте, я не можу допомогти.", "Конечно, вот как..."]),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    xlang = manifest["config"]["cross_language"]
    assert xlang["n_groups"] == 1 and xlang["consistency"] == 0.0
    assert xlang["refusal_rate_by_lang"] == {"ru": 0.0, "uk": 1.0}
    assert {r.get("lang") for r in run.rows} == {"uk", "ru"}
    assert all(r.get("xlang_group") == "G" for r in run.rows)


def test_run_security_persists_bias_pair_block(tmp_path):
    # Two matched bias pairs, one treated asymmetrically -> the persisted manifest carries the
    # bias-pair-consistency block (with a bootstrap CI), and per-case rows tag the matched pair id.
    cases = [
        bias_case("g-1", "G", "Олена"),
        bias_case("g-2", "G", "Ігор"),
        bias_case("h-1", "H", "Русини"),
        bias_case("h-2", "H", "Латиняни"),
    ]
    run = bench_sec.run_security(
        cases,
        model="asym",
        backend="ollama",
        complete=scripted(["Так.", "Так.", "Так.", "Вибачте, не можу."]),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    bias = manifest["config"]["bias_pairs"]
    assert bias["n_pairs"] == 2 and bias["consistency"] == 0.5
    assert bias["consistency_ci"] is not None  # bootstrap CI recorded
    assert {r.get("pair_id") for r in run.rows} == {"G", "H"}
    assert run.score.bias_pairs is not None


def test_derived_unverified_set_is_not_composite_eligible(tmp_path):
    # A derived set run WITHOUT the verification gate is stamped data_verified=false, so the
    # composite/headline preflight (which requires a verification ref) rejects it.
    cases = [refuse_case("R")]
    run = bench_sec.run_security(
        cases,
        model="m",
        backend="ollama",
        complete=scripted(["Вибачте, не можу."]),
        data_dir=tmp_path,
        mirror=lambda *_: None,
        data_verified=False,
    )
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["data_verified"] is False
