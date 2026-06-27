"""Tier-1 public screen (public screen + verified gold-set ledger Belebele wiring): tracks, parsing, coverage -- faked."""

import pytest

from llb.executor.vram import VramNotReclaimed
from llb.screen.public import (
    GENERATION_TASKS,
    LOGPROB_TASKS,
    TRACK_GENERATION,
    TRACK_LOGPROB,
    assert_single_track,
    build_lm_eval_command,
    format_screen,
    parse_results,
    run_screen,
    run_screen_isolated,
    screen_score,
    select_finalists,
    select_tasks,
    supports_logprobs,
)


def _gpu(temp=40):
    return [
        {"index": 0, "temp_c": temp, "power_w": 100.0, "sm_clock_mhz": 2000, "mem_clock_mhz": 9000}
    ]


def _fake_report(model="m", track=TRACK_LOGPROB):
    return {
        "model": model,
        "backend": "vllm",
        "track": track,
        "requested_tasks": ["t"],
        "results": [{"task": "t", "metric": "acc", "score": 0.5}],
        "covered": ["t"],
        "missing": [],
        "complete": True,
    }


def test_run_screen_isolated_gates_vram_for_vllm():
    report, iso = run_screen_isolated(
        "vllm",
        lambda: _fake_report(),
        vram_reader=lambda: 1000,  # constant -> reclaimed within tolerance
        gpu_sampler=lambda: _gpu(40),
        sleep=lambda _s: None,
    )
    assert report["model"] == "m"
    assert iso["vram_residual_mb"] == 0 and iso["cooldown"]["capped"] is False


def test_run_screen_isolated_skips_vram_gate_for_ollama():
    _report, iso = run_screen_isolated(
        "ollama",
        lambda: _fake_report(track=TRACK_GENERATION),
        vram_reader=lambda: 9000,  # would trip the gate if it applied
        gpu_sampler=lambda: _gpu(40),
        sleep=lambda _s: None,
    )
    assert iso["vram_residual_mb"] is None  # Ollama keeps weights warm -> never gated


def test_run_screen_isolated_aborts_on_unreclaimed_vram():
    reads = iter([1000] + [9000] * 100)  # baseline low, then never returns
    with pytest.raises(VramNotReclaimed):
        run_screen_isolated(
            "vllm",
            lambda: _fake_report(),
            vram_reader=lambda: next(reads),
            gpu_sampler=lambda: _gpu(40),
            sleep=lambda _s: None,
        )


def _report(model, track, scores):
    return {
        "model": model,
        "backend": "x",
        "track": track,
        "requested_tasks": list(scores),
        "results": [{"task": t, "metric": "acc", "score": s} for t, s in scores.items()],
        "covered": list(scores),
        "missing": [],
        "complete": True,
    }


def test_screen_score_is_mean_of_task_scores():
    assert screen_score(_report("m", TRACK_LOGPROB, {"a": 0.6, "b": 0.8})) == 0.7


def test_select_finalists_top_n_per_track_deterministic():
    reports = [
        _report("hi", TRACK_LOGPROB, {"t": 0.9}),
        _report("lo", TRACK_LOGPROB, {"t": 0.3}),
        _report("mid", TRACK_LOGPROB, {"t": 0.6}),
        _report("gen", TRACK_GENERATION, {"t": 0.5}),
    ]
    # per-track top-2 by mean score; tracks processed in sorted order (generation < logprob).
    assert select_finalists(reports, top_n=2) == ["gen", "hi", "mid"]


FAKE_RESULTS = {
    "results": {
        "belebele_ukr_Cyrl": {"acc,none": 0.62, "acc_stderr,none": 0.01},
        "arc_uk": {"acc,none": 0.55, "acc_stderr,none": 0.01},
        "hellaswag_uk": {"acc,none": 0.50, "acc_stderr,none": 0.01},
        "m_mmlu_uk": {"acc,none": 0.45, "acc_stderr,none": 0.01},
        "global_piqa_prompted_ukr_cyrl": {"exact_match,none": 0.40, "f1,none": 0.55},
    }
}


def test_supports_logprobs_only_vllm():
    assert supports_logprobs("vllm") is True
    assert supports_logprobs("ollama") is False


def test_select_tasks_per_track():
    track, tasks = select_tasks("vllm")
    assert track == TRACK_LOGPROB
    assert set(LOGPROB_TASKS).issubset(tasks) and set(GENERATION_TASKS).issubset(tasks)

    track, tasks = select_tasks("ollama")
    assert track == TRACK_GENERATION
    assert all(t not in tasks for t in LOGPROB_TASKS)  # no MCQ-by-loglikelihood without logprobs


def test_select_tasks_deduplicates_defaults_from_extra_tasks():
    _track, tasks = select_tasks("vllm", ["belebele_ukr_Cyrl", "custom", "custom"])
    assert tasks.count("belebele_ukr_Cyrl") == 1  # default not duplicated by an --extra repeat
    assert tasks.count("custom") == 1


def test_build_lm_eval_command():
    cmd = build_lm_eval_command("m", "http://h:8000/v1", ["t1", "t2"], "/out", limit=5)
    assert cmd[:3] == ["lm_eval", "--model", "local-completions"]
    assert "t1,t2" in cmd and "--limit" in cmd and "5" in cmd
    assert any("base_url=http://h:8000/v1/completions" in a for a in cmd)


def test_build_lm_eval_command_tokenizer_is_track_aware():
    # logprob (MCQ/loglikelihood) needs the model's HF tokenizer to compute context lengths.
    logprob = build_lm_eval_command("org/m", "http://h/v1", ["t"], "/o", track=TRACK_LOGPROB)
    assert any("tokenizer=org/m" in a for a in logprob)
    # generation (Ollama tag) disables the local tokenizer (a tag is not a valid HF repo id).
    gen = build_lm_eval_command("llama3.2:3b", "http://h/v1", ["t"], "/o", track=TRACK_GENERATION)
    assert any("tokenizer_backend=None" in a for a in gen)


def test_parse_results_scores_and_flags_missing_coverage():
    report = parse_results(
        FAKE_RESULTS,
        ["belebele_ukr_Cyrl", "global_piqa_prompted_ukr_cyrl", "absent_task"],
        model="m",
        backend="vllm",
        track=TRACK_LOGPROB,
    )
    by_task = {r["task"]: (r["metric"], r["score"]) for r in report["results"]}
    assert by_task["belebele_ukr_Cyrl"] == ("acc", 0.62)
    assert by_task["global_piqa_prompted_ukr_cyrl"][0] == "exact_match"  # preferred over f1
    assert report["missing"] == ["absent_task"] and report["complete"] is False


def test_parse_results_never_uses_stderr_as_the_metric():
    report = parse_results(
        {"results": {"task": {"custom_stderr,none": 0.01}}},
        ["task"],
        model="m",
        backend="vllm",
        track=TRACK_LOGPROB,
    )
    assert report["results"] == [] and report["missing"] == ["task"]


def test_run_screen_with_injected_runner():
    report = run_screen(
        "m", "vllm", "http://h:8000/v1", runner=lambda _cmd: FAKE_RESULTS, output_dir=None
    )
    assert report["track"] == TRACK_LOGPROB
    assert report["complete"] is True  # every default task present in the fake results
    assert {r["task"] for r in report["results"]} == set(LOGPROB_TASKS) | set(GENERATION_TASKS)


def test_assert_single_track_rejects_mixed():
    log = {"track": TRACK_LOGPROB}
    gen = {"track": TRACK_GENERATION}
    assert assert_single_track([log, log]) == TRACK_LOGPROB
    with pytest.raises(ValueError, match="across screen tracks"):
        assert_single_track([log, gen])


def test_format_screen_is_ascii_with_coverage():
    rep = parse_results(
        FAKE_RESULTS,
        ["belebele_ukr_Cyrl", "global_piqa_prompted_ukr_cyrl", "absent_task"],
        model="m",
        backend="vllm",
        track=TRACK_LOGPROB,
    )
    table = format_screen([rep])
    assert table.isascii() and "coverage" in table and "PARTIAL" in table
