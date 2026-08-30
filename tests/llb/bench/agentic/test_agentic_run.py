"""Tests for agentic run."""

import json
from pathlib import Path
from llb.bench.agentic.model import (
    STATUS_COMPLETED,
    AgenticTask,
    Episode,
)
from llb.bench.agentic.run import load_tasks_file, run_agentic
from llb.bench.agentic.trajectory import _trajectory_records, trajectory_quality
from llb.bench import tool_world as tw
from llb.scoring.aggregate import TIER_AGENTIC
from tests.llb.bench.agentic.test_agentic import fake_judge, scripted


def test_run_agentic_completion_rate_and_persist(tmp_path):
    tasks = [
        AgenticTask(
            "a",
            "calc+write",
            success=[{"kind": "file_equals", "path": "result.txt", "value": "84"}],
        ),
        AgenticTask("b", "db", success=[{"kind": "db_equals", "key": "capital", "value": "Київ"}]),
    ]
    # global sequence of complete() calls across both episodes, in order
    complete = scripted(
        [
            '{"name":"calculator","arguments":{"expression":"12 * (3 + 4)"}}',
            '{"name":"write_file","arguments":{"path":"result.txt","content":"84"}}',
            '{"name":"finish","arguments":{"answer":"done"}}',
            '{"name":"db_set","arguments":{"key":"capital","value":"Київ"}}',
            '{"name":"finish","arguments":{"answer":"done"}}',
        ]
    )
    run = run_agentic(
        tasks,
        model="m",
        backend="ollama",
        complete=complete,
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.result.tier == TIER_AGENTIC
    assert run.result.objective_score == 1.0
    assert run.completion_ci is not None
    assert run.mean_tool_calls == 1.5  # (2 + 1) / 2
    assert run.paths is not None and "agentic" in run.paths["manifest"]


def test_run_agentic_failing_model():
    tasks = load_tasks_file("samples/benchmarks/agentic_tasks_uk.json")
    run = run_agentic(
        tasks,
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":""}}',
        persist=False,
    )
    assert run.result.objective_score == 0.0  # finishes immediately, no env changes


def test_run_agentic_reports_meter_throughput(tmp_path):
    import json
    from pathlib import Path

    from llb.bench.common_backend import ThroughputMeter

    meter = ThroughputMeter()
    meter.completion_tokens, meter.generation_s, meter.calls = 100, 4.0, 4  # 25 tok/s
    tasks = [AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "x"}])]
    run = run_agentic(
        tasks,
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":"x"}}',
        data_dir=tmp_path,
        mirror=lambda *_: None,
        meter=meter,
    )
    assert run.result.tokens_per_s == 25.0  # real throughput flows onto the board row
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["tokens_per_s"] == 25.0


def test_load_tasks_file_and_from_record_coerces_success():
    tasks = load_tasks_file("samples/benchmarks/agentic_tasks_uk.json")
    assert len(tasks) == 4 and all(t.success for t in tasks)
    one = AgenticTask.from_record(
        {"id": "x", "prompt": "p", "success": {"kind": "answer_contains", "value": "y"}}
    )
    assert isinstance(one.success, list) and len(one.success) == 1


def test_agentic_case_row_shape():
    tasks = [AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "hi"}])]
    run = run_agentic(
        tasks, model="m", backend="ollama", complete=lambda _: "hi there", persist=False
    )
    row = run.rows[0]
    assert row["item_id"] == "a" and row["success"] == 1.0 and json.dumps(row)


def test_trajectory_quality_averages_the_two_signals():
    assert trajectory_quality({"faithfulness": 0.8, "answer_relevancy": 0.6}) == 0.7


def test_trajectory_records_carry_observations_as_context():
    task = AgenticTask("t", "обчисли", success=[])
    episode = Episode(
        success=False,
        status=STATUS_COMPLETED,
        n_steps=1,
        n_tool_calls=1,
        answer="готово",
        world=tw.ToolWorld(),
        transcript=[("calculator", {"expression": "2+2"}, "4")],
    )
    recs = _trajectory_records([task], [episode])
    assert recs[0]["answer"] == "готово"
    assert "обчисли" in recs[0]["question"]
    assert recs[0]["contexts"] == ['calculator({"expression": "2+2"}) -> 4']


def test_agentic_gated_judge_trusted_records_quality(tmp_path):
    tasks = [
        AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "x"}]),
        AgenticTask("b", "p", success=[{"kind": "answer_contains", "value": "x"}]),
    ]
    # the model finishes with an empty answer -> objective completion stays 0
    run = run_agentic(
        tasks,
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":""}}',
        judge_model="judge",
        judge_rho=0.7,  # >= 0.6 -> trusted
        judge_scorer=fake_judge(0.8, 0.6),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.judge_trusted is True
    assert run.trajectory_quality == 0.7 and run.trajectory_quality_ci is not None
    assert all(row["trajectory_quality"] == 0.7 for row in run.rows)
    # the headline stays OBJECTIVE completion-rate -- trajectory quality is NOT folded in
    assert run.result.objective_score == 0.0


def test_agentic_gated_judge_below_threshold_is_demoted():
    tasks = [AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "x"}])]
    run = run_agentic(
        tasks,
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":"x"}}',
        judge_model="judge",
        judge_rho=0.3,  # < 0.6 -> demoted
        judge_scorer=fake_judge(0.9, 0.9),
        persist=False,
    )
    assert run.judge_trusted is False and run.trajectory_quality is None
    assert "trajectory_quality" not in run.rows[0]


def test_agentic_no_judge_is_objective_only():
    tasks = [AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "x"}])]
    run = run_agentic(
        tasks,
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":"x"}}',
        persist=False,
    )
    assert run.judge_trusted is False and run.trajectory_quality is None
    assert run.judge_reason == "no judge configured"


# --- loop policy on the scored run --------------------------------------------------------------


REPEATED_CALL_SCRIPT = [
    '{"name":"db_set","arguments":{"key":"capital","value":"Київ"}}',
    '{"name":"db_set","arguments":{"key":"capital","value":"Київ"}}',
    '{"name":"finish","arguments":{"answer":"done"}}',
]


def _repeat_task():
    return AgenticTask(
        "r", "db", success=[{"kind": "db_equals", "key": "capital", "value": "Київ"}]
    )


def _recording_complete(outputs):
    """A scripted completion that also keeps every prompt it was sent, in order."""
    prompts: list[str] = []
    it = iter(outputs)

    def complete(prompt):
        prompts.append(prompt)
        return next(it)

    return complete, prompts


def test_run_agentic_threads_a_non_default_loop_policy_onto_the_loop_harness():
    """A recommended cell is RUN here, not merely re-swept: the noop policy changes the episode."""
    from llb.bench.agentic.loop_policy import REPEATED_NOOP, LoopPolicy

    run = run_agentic(
        [_repeat_task()],
        model="m",
        backend="ollama",
        complete=scripted(REPEATED_CALL_SCRIPT),
        loop_policy=LoopPolicy(repeated_call=REPEATED_NOOP),
        persist=False,
    )

    assert run.episodes[0].n_repeated_calls == 1
    assert (
        run.episodes[0].n_repeated_noops == 1
    )  # the second identical call never reached the world


def test_run_agentic_default_loop_policy_leaves_the_episode_alone():
    run = run_agentic(
        [_repeat_task()],
        model="m",
        backend="ollama",
        complete=scripted(REPEATED_CALL_SCRIPT),
        persist=False,
    )

    assert run.episodes[0].n_repeated_calls == 1
    assert run.episodes[0].n_repeated_noops == 0  # `allow` executes it


def test_default_cell_prompts_are_byte_identical_with_and_without_an_explicit_policy():
    """Threading the policy must not move the shipped run's prompts by a single byte."""
    from llb.bench.agentic.loop_policy import LoopPolicy

    implicit_complete, implicit_prompts = _recording_complete(REPEATED_CALL_SCRIPT)
    explicit_complete, explicit_prompts = _recording_complete(REPEATED_CALL_SCRIPT)
    common = {"model": "m", "backend": "ollama", "persist": False}

    run_agentic([_repeat_task()], complete=implicit_complete, **common)
    run_agentic([_repeat_task()], complete=explicit_complete, loop_policy=LoopPolicy(), **common)

    assert implicit_prompts == explicit_prompts


def test_manifest_records_the_loop_cell_that_ran(tmp_path):
    from llb.bench.agentic.loop_policy import (
        MALFORMED_STRICT,
        REPEAT_FEEDBACK_UK,
        REPEATED_NOOP,
        LoopPolicy,
    )

    run = run_agentic(
        [_repeat_task()],
        model="m",
        backend="ollama",
        complete=scripted(REPEATED_CALL_SCRIPT),
        loop_policy=LoopPolicy(
            malformed_call=MALFORMED_STRICT,
            repeated_call=REPEATED_NOOP,
            repeat_feedback=REPEAT_FEEDBACK_UK,
        ),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )

    assert run.paths is not None
    config = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))["config"]
    assert config["malformed_call_policy"] == MALFORMED_STRICT
    assert config["repeated_call_policy"] == REPEATED_NOOP
    assert config["repeat_feedback_variant"] == REPEAT_FEEDBACK_UK
    assert config["loop_policy_supported"] is True


def test_manifest_records_the_shipped_cell_when_no_policy_is_passed(tmp_path):
    from llb.bench.agentic.loop_policy import (
        DEFAULT_MALFORMED_POLICY,
        DEFAULT_REPEAT_FEEDBACK,
        DEFAULT_REPEATED_CALL_POLICY,
    )

    run = run_agentic(
        [_repeat_task()],
        model="m",
        backend="ollama",
        complete=scripted(REPEATED_CALL_SCRIPT),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )

    assert run.paths is not None
    config = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))["config"]
    assert config["malformed_call_policy"] == DEFAULT_MALFORMED_POLICY
    assert config["repeated_call_policy"] == DEFAULT_REPEATED_CALL_POLICY
    assert config["repeat_feedback_variant"] == DEFAULT_REPEAT_FEEDBACK
