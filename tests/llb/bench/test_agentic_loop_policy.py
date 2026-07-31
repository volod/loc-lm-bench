"""Agent-loop policy branches, paired grid, recommendation, and persistence."""

import json
from pathlib import Path

import pytest

from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    MALFORMED_REPAIR_ONCE,
    MALFORMED_STRICT,
    REPEATED_ALLOW,
    REPEATED_NOOP,
    REPEATED_NOOP_OBSERVATION,
    LoopPolicy,
)
from llb.bench.agentic.model import STATUS_COMPLETED, AgenticTask
from llb.bench.agentic_loop_policy import policy_grid, run_agentic_loop_policy
from llb.bench.agentic_loop_policy_report import METRICS
from llb.scoring.tool_calls import parse_tool_call, parse_tool_call_detailed


def task(task_id: str = "t") -> AgenticTask:
    return AgenticTask(
        task_id,
        "write the result",
        success=[{"kind": "file_equals", "path": "result.txt", "value": "ok"}],
    )


def scripted(outputs: list[str]):
    iterator = iter(outputs)
    return lambda _prompt: next(iterator)


def test_detailed_parser_separates_plain_answer_from_malformed_call():
    plain = parse_tool_call_detailed("ordinary final answer")
    malformed = parse_tool_call_detailed('{"name":"write_file","arguments":')
    assert plain.call is None and plain.attempted is False
    assert malformed.call is None and malformed.attempted is True
    assert "invalid JSON" in (malformed.error or "")
    assert parse_tool_call('{"name":"write_file","arguments":') is None


def test_default_loop_policy_reproduces_implicit_legacy_behavior():
    outputs = [
        '{"name":"write_file","arguments":{"path":"result.txt","content":"ok"}}',
        '{"name":"finish","arguments":{"answer":"done"}}',
    ]
    implicit = run_episode(task(), scripted(outputs))
    explicit = run_episode(
        task(),
        scripted(outputs),
        loop_policy=LoopPolicy(MALFORMED_ANSWER, REPEATED_ALLOW),
    )
    assert (
        implicit.success,
        implicit.status,
        implicit.n_steps,
        implicit.n_tool_calls,
        implicit.answer,
        implicit.transcript,
    ) == (
        explicit.success,
        explicit.status,
        explicit.n_steps,
        explicit.n_tool_calls,
        explicit.answer,
        explicit.transcript,
    )


def test_answer_policy_preserves_malformed_as_final_answer_behavior():
    episode = run_episode(
        task(),
        lambda _prompt: '{"name":"write_file","arguments":',
        loop_policy=LoopPolicy(malformed_call=MALFORMED_ANSWER),
    )
    assert episode.status == STATUS_COMPLETED
    assert episode.success is False
    assert episode.n_steps == 1 and episode.n_malformed_calls == 1
    assert episode.n_repair_attempts == 0


def test_repair_once_recovers_an_unreadable_completion_and_counts_its_cost():
    prompts: list[str] = []
    outputs = iter(
        [
            '{"name":"write_file","arguments":',
            '{"name":"write_file","arguments":{"path":"result.txt","content":"ok"}}',
            '{"name":"finish","arguments":{"answer":"done"}}',
        ]
    )

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    episode = run_episode(
        task(),
        complete,
        loop_policy=LoopPolicy(malformed_call=MALFORMED_REPAIR_ONCE),
    )
    assert episode.success is True and episode.n_steps == 2
    assert episode.n_malformed_calls == 1 and episode.n_repair_attempts == 1
    assert episode.n_model_calls == 3
    assert "Parse error:" in prompts[1] and '"write_file"' in prompts[1]
    assert episode.telemetry.n_repair_prompts == 1


def test_strict_policy_records_feedback_and_continues_without_executing():
    episode = run_episode(
        task(),
        scripted(
            [
                '{"name":"write_file","arguments":',
                '{"name":"write_file","arguments":{"path":"result.txt","content":"ok"}}',
                '{"name":"finish","arguments":{"answer":"done"}}',
            ]
        ),
        loop_policy=LoopPolicy(malformed_call=MALFORMED_STRICT),
    )
    assert episode.success is True and episode.n_steps == 3
    assert episode.n_malformed_calls == 1 and episode.n_tool_calls == 1
    assert len(episode.transcript) == 1


def test_repeated_noop_is_recorded_but_does_not_execute_again():
    outputs = [
        '{"name":"db_get","arguments":{"key":"missing"}}',
        '{"name":"db_get","arguments":{"key":"missing"}}',
        '{"name":"finish","arguments":{"answer":"done"}}',
    ]
    episode = run_episode(
        AgenticTask("t", "inspect", success=[{"kind": "answer_contains", "value": "done"}]),
        scripted(outputs),
        loop_policy=LoopPolicy(repeated_call=REPEATED_NOOP),
    )
    assert episode.success is True and episode.n_tool_calls == 2
    assert episode.n_repeated_calls == 1 and episode.n_repeated_noops == 1
    assert episode.transcript[-1][2] == REPEATED_NOOP_OBSERVATION


def test_repeated_allow_is_counted_without_being_suppressed():
    outputs = [
        '{"name":"db_get","arguments":{"key":"missing"}}',
        '{"name":"db_get","arguments":{"key":"missing"}}',
        '{"name":"finish","arguments":{"answer":"done"}}',
    ]
    episode = run_episode(
        AgenticTask("t", "inspect", success=[{"kind": "answer_contains", "value": "done"}]),
        scripted(outputs),
        loop_policy=LoopPolicy(repeated_call=REPEATED_ALLOW),
    )
    assert episode.n_repeated_calls == 1 and episode.n_repeated_noops == 0


def test_grid_requires_the_exact_legacy_baseline():
    with pytest.raises(SystemExit, match="must include baseline"):
        policy_grid([4], [MALFORMED_STRICT], [REPEATED_NOOP])
    cells = policy_grid(
        [6, 4],
        [MALFORMED_ANSWER, MALFORMED_STRICT],
        [REPEATED_ALLOW, REPEATED_NOOP],
    )
    assert sum(cell.is_baseline for cell in cells) == 1


def test_sweep_pairs_every_cell_and_persists_comparison_artifacts(tmp_path: Path):
    tasks = [
        AgenticTask(
            f"t{i}",
            "finish",
            success=[{"kind": "answer_contains", "value": "done"}],
        )
        for i in range(8)
    ]
    run = run_agentic_loop_policy(
        tasks,
        model="m",
        backend="ollama",
        complete=lambda _prompt: '{"name":"finish","arguments":{"answer":"done"}}',
        max_steps=[4, 6],
        malformed_policies=[MALFORMED_ANSWER, MALFORMED_REPAIR_ONCE],
        repeated_call_policies=[REPEATED_ALLOW, REPEATED_NOOP],
        data_dir=tmp_path,
        mirror=lambda *_args: None,
    )
    assert len(run.reports) == 8
    assert all(set(report.paired) == set(METRICS) for report in run.reports)
    assert run.recommendation["changes_shipped_defaults"] is False
    assert run.recommendation["max_steps"] == 6
    for report in run.reports:
        assert report.paths is not None
        run_dir = Path(report.paths["manifest"]).parent
        assert (run_dir / "comparison.md").is_file()
        recommendation = json.loads((run_dir / "recommendation.json").read_text())
        assert recommendation["malformed_call_policy"] == MALFORMED_ANSWER
        manifest = json.loads(Path(report.paths["manifest"]).read_text())
        assert set(manifest["config"]["paired_vs_baseline"]) == set(METRICS)


def test_recommendation_changes_only_after_a_positive_standard_verdict():
    tasks = [
        AgenticTask(
            f"t{i}",
            "finish",
            success=[{"kind": "answer_contains", "value": "done"}],
        )
        for i in range(8)
    ]

    def complete(prompt: str) -> str:
        if "previous response looked like a tool call" in prompt:
            return '{"name":"finish","arguments":{"answer":"done"}}'
        return '{"name":"finish","arguments":'

    run = run_agentic_loop_policy(
        tasks,
        model="m",
        backend="ollama",
        complete=complete,
        max_steps=[6],
        malformed_policies=[MALFORMED_ANSWER, MALFORMED_REPAIR_ONCE],
        repeated_call_policies=[REPEATED_ALLOW],
        persist=False,
    )
    assert run.recommendation["changes_shipped_defaults"] is True
    assert run.recommendation["malformed_call_policy"] == MALFORMED_REPAIR_ONCE
    assert run.recommendation["verdict"] == "separated"
