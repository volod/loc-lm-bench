"""Tests for harness integrations."""

import pytest
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import (
    HARNESS_LANGGRAPH,
    HARNESS_LOOP,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    AgenticTask,
)
from llb.bench.agentic.run import run_agentic
from llb.bench import tool_world as tw
from llb.bench.harness import crewai as crewai_harness
from llb.bench.harness import crewai_runtime
from llb.bench.harness import langgraph as lg
from llb.board.harnesses import harness_comparison, load_agentic_harness_records
from tests.llb.bench.harness.test_harness import (
    SUCCESS_SCRIPT,
    fake_crew_runner,
    loop_script,
    scripted,
    success_task,
    two_tasks,
)


def test_recording_executor_runs_and_records():
    world = tw.ToolWorld.from_setup({"db": {"k": "v"}})
    transcript = []
    execute = crewai_runtime.make_recording_executor(world, transcript)
    assert execute("db_get", {"key": "k"}) == "v"
    assert execute("db_set", {"key": "k2", "value": "v2"}) == tw.OBS_OK
    assert world.db["k2"] == "v2"
    assert [name for name, _a, _o in transcript] == ["db_get", "db_set"]


def test_crew_tool_specs_excludes_finish():
    specs = crewai_runtime.crew_tool_specs(tw.tool_catalog(), lambda n, a: "")
    names = {s.name for s in specs}
    assert tw.FINISH not in names and tw.WRITE_FILE in names


def test_crewai_harness_with_fake_crew():
    harness = crewai_harness.make_crewai_harness(fake_crew_runner)
    ep = harness(success_task(), scripted([]), tw.tool_catalog(), max_steps=6)
    assert ep.success is True and ep.status == STATUS_COMPLETED
    assert ep.n_tool_calls == 2 and ep.answer == "готово"
    assert ep.context_policy_supported is False
    assert ep.telemetry.max_prompt_chars > 0


def test_crewai_harness_marks_policy_unsupported():
    from llb.bench.agentic.context_policy import ContextPolicy, POLICY_OBSERVATION_CAP

    harness = crewai_harness.make_crewai_harness(fake_crew_runner)
    ep = harness(
        success_task(),
        scripted([]),
        tw.tool_catalog(),
        policy=ContextPolicy(name=POLICY_OBSERVATION_CAP),
    )
    assert ep.context_policy_supported is False
    assert ep.success is True


def test_episode_from_outcome_incomplete_when_not_finished():
    task = AgenticTask("t", "p", success=[{"kind": "answer_contains", "value": "x"}])
    world = tw.ToolWorld.from_setup({})
    outcome = crewai_runtime.CrewOutcome(answer="", finished=False)
    ep = crewai_runtime.episode_from_outcome(task, world, outcome)
    assert ep.status == STATUS_INCOMPLETE and ep.success is False
    assert ep.context_policy_supported is False


def test_run_agentic_records_harness_in_manifest(tmp_path):
    import json
    from pathlib import Path

    run = run_agentic(
        two_tasks(),
        model="m",
        backend="ollama",
        complete=loop_script(),
        harness_name=HARNESS_LANGGRAPH,
        harness=lg.step_graph_pure,
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.paths is not None
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["harness"] == HARNESS_LANGGRAPH
    assert manifest["config"]["context_policy"] == "full"
    assert manifest["config"]["context_policy_supported"] is True
    assert manifest["config"]["mean_max_prompt_tokens"] > 0
    assert run.result.objective_score == 1.0


def test_langgraph_applies_observation_cap_like_loop():
    from llb.bench.agentic.context_policy import (
        POLICY_FULL,
        POLICY_OBSERVATION_CAP,
        ContextPolicy,
    )
    from llb.backends.context_budget import unbounded_budget

    catalog = tw.tool_catalog()
    # One fat observation then finish: observation_cap must trim what the next prompt sees.
    fat = "X" * 2000
    task = AgenticTask(
        "t",
        "read then finish",
        setup={"files": {"big.txt": fat}},
        success=[{"kind": "answer_contains", "value": "ok"}],
    )
    script = [
        '{"name":"read_file","arguments":{"path":"big.txt"}}',
        '{"name":"finish","arguments":{"answer":"ok"}}',
    ]
    policy = ContextPolicy(name=POLICY_OBSERVATION_CAP, observation_cap_chars=100)
    budget = unbounded_budget()
    loop_ep = run_episode(task, scripted(script), catalog=catalog, policy=policy, budget=budget)
    graph_ep = lg.step_graph_pure(task, scripted(script), catalog, policy=policy, budget=budget)
    full_ep = run_episode(
        task,
        scripted(script),
        catalog=catalog,
        policy=ContextPolicy(name=POLICY_FULL),
        budget=budget,
    )
    assert loop_ep.context_policy_supported and graph_ep.context_policy_supported
    assert loop_ep.telemetry.n_trimmed_observations == 1
    assert graph_ep.telemetry.n_trimmed_observations == 1
    assert loop_ep.telemetry.max_prompt_chars == graph_ep.telemetry.max_prompt_chars
    assert (
        loop_ep.telemetry.model_input_prompt_chars
        == graph_ep.telemetry.model_input_prompt_chars
        > 0
    )
    assert loop_ep.telemetry.max_prompt_chars < full_ep.telemetry.max_prompt_chars


def test_harness_comparison_ranks_one_model_across_harnesses(tmp_path):
    # loop harness: both tasks succeed
    run_agentic(
        two_tasks(),
        model="m",
        backend="ollama",
        complete=loop_script(),
        harness_name=HARNESS_LOOP,
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    # a "langgraph" run (same pure nodes) that fails (model finishes empty immediately)
    run_agentic(
        two_tasks(),
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":""}}',
        harness_name=HARNESS_LANGGRAPH,
        harness=lg.step_graph_pure,
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    records = load_agentic_harness_records(tmp_path)
    assert {(r.model, r.harness) for r in records} == {("m", "loop"), ("m", "langgraph")}
    rows, table, harnesses = harness_comparison(tmp_path, "m")
    assert {row["model"] for row in rows} == {"loop", "langgraph"}
    # the loop (1.0) outranks the failing langgraph run (0.0)
    top = next(row for row in rows if row["rank"] == 1)
    assert top["model"] == "loop"
    assert "policy:" in table
    assert "prompt-tok" in table
    assert "applied" in table
    assert "context-policy=full" in table


def test_harness_comparison_holds_context_policy_fixed(tmp_path):
    from llb.bench.agentic.context_policy import ContextPolicy, POLICY_OBSERVATION_CAP

    run_agentic(
        two_tasks()[:1],
        model="m",
        backend="ollama",
        complete=loop_script(),
        harness_name=HARNESS_LOOP,
        policy=ContextPolicy(),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    run_agentic(
        two_tasks()[:1],
        model="m",
        backend="ollama",
        complete=loop_script(),
        harness_name=HARNESS_LOOP,
        policy=ContextPolicy(name=POLICY_OBSERVATION_CAP),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    run_agentic(
        two_tasks()[:1],
        model="m",
        backend="ollama",
        complete=loop_script(),
        harness_name=HARNESS_LANGGRAPH,
        harness=lg.step_graph_pure,
        policy=ContextPolicy(name=POLICY_OBSERVATION_CAP),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    records = load_agentic_harness_records(tmp_path)
    assert len(records) == 3  # two policies for loop + one for langgraph
    _rows, table, harnesses = harness_comparison(tmp_path, "m")
    # observation_cap covers two harnesses; full covers one -> observation_cap wins
    assert set(harnesses) == {HARNESS_LOOP, HARNESS_LANGGRAPH}
    assert "context-policy=observation_cap" in table
    assert "observation_cap" in table
    _rows2, table2, harnesses2 = harness_comparison(tmp_path, "m", context_policy="full")
    assert harnesses2 == [HARNESS_LOOP]
    assert "context-policy=full" in table2


def test_harness_comparison_marks_crewai_policy_unsupported(tmp_path):
    run_agentic(
        two_tasks()[:1],
        model="m",
        backend="ollama",
        complete=loop_script(),
        harness_name="crewai",
        harness=crewai_harness.make_crewai_harness(fake_crew_runner),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    records = load_agentic_harness_records(tmp_path)
    assert len(records) == 1
    assert records[0].context_policy_supported is False
    assert records[0].context_policy == "full"
    assert records[0].mean_max_prompt_tokens > 0
    _rows, table, _ = harness_comparison(tmp_path, "m")
    assert "full*" in table
    assert "no" in table


@pytest.mark.slow
def test_real_langgraph_harness_matches_loop():
    pytest.importorskip("langgraph")
    catalog = tw.tool_catalog()
    loop_ep = run_episode(success_task(), scripted(SUCCESS_SCRIPT), catalog=catalog)
    graph_ep = lg.langgraph_harness(success_task(), scripted(SUCCESS_SCRIPT), catalog)
    assert (
        graph_ep.success,
        graph_ep.status,
        graph_ep.n_steps,
        graph_ep.n_tool_calls,
        graph_ep.answer,
    ) == (loop_ep.success, loop_ep.status, loop_ep.n_steps, loop_ep.n_tool_calls, loop_ep.answer)
    assert graph_ep.world.files == loop_ep.world.files
    assert graph_ep.telemetry.max_prompt_chars == loop_ep.telemetry.max_prompt_chars
    assert graph_ep.context_policy_supported is True
