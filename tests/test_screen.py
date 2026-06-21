"""Tier-1 public screen (M3.1 + M3.9 Belebele wiring): tracks, parsing, coverage -- faked."""

import pytest

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
    select_tasks,
    supports_logprobs,
)

FAKE_RESULTS = {
    "results": {
        "belebele_ukr_Cyrl": {"acc,none": 0.62, "acc_stderr,none": 0.01},
        "squad_uk": {"exact_match,none": 0.40, "f1,none": 0.55},
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


def test_build_lm_eval_command():
    cmd = build_lm_eval_command("m", "http://h:8000/v1", ["t1", "t2"], "/out", limit=5)
    assert cmd[:3] == ["lm_eval", "--model", "local-completions"]
    assert "t1,t2" in cmd and "--limit" in cmd and "5" in cmd
    assert any("base_url=http://h:8000/v1/completions" in a for a in cmd)


def test_parse_results_scores_and_flags_missing_coverage():
    report = parse_results(
        FAKE_RESULTS,
        ["belebele_ukr_Cyrl", "squad_uk", "absent_task"],
        model="m",
        backend="vllm",
        track=TRACK_LOGPROB,
    )
    by_task = {r["task"]: (r["metric"], r["score"]) for r in report["results"]}
    assert by_task["belebele_ukr_Cyrl"] == ("acc", 0.62)
    assert by_task["squad_uk"][0] == "exact_match"  # preferred over f1
    assert report["missing"] == ["absent_task"] and report["complete"] is False


def test_run_screen_with_injected_runner():
    report = run_screen(
        "m", "vllm", "http://h:8000/v1", runner=lambda _cmd: FAKE_RESULTS, output_dir=None
    )
    assert report["track"] == TRACK_LOGPROB
    assert report["complete"] is True  # both default tasks present
    assert {r["task"] for r in report["results"]} == {"belebele_ukr_Cyrl", "squad_uk"}


def test_assert_single_track_rejects_mixed():
    log = {"track": TRACK_LOGPROB}
    gen = {"track": TRACK_GENERATION}
    assert assert_single_track([log, log]) == TRACK_LOGPROB
    with pytest.raises(ValueError, match="across screen tracks"):
        assert_single_track([log, gen])


def test_format_screen_is_ascii_with_coverage():
    rep = parse_results(
        FAKE_RESULTS,
        ["belebele_ukr_Cyrl", "squad_uk", "absent_task"],
        model="m",
        backend="vllm",
        track=TRACK_LOGPROB,
    )
    table = format_screen([rep])
    assert table.isascii() and "coverage" in table and "PARTIAL" in table
