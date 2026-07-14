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
from test_harness import (
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


def test_episode_from_outcome_incomplete_when_not_finished():
    task = AgenticTask("t", "p", success=[{"kind": "answer_contains", "value": "x"}])
    world = tw.ToolWorld.from_setup({})
    outcome = crewai_runtime.CrewOutcome(answer="", finished=False)
    ep = crewai_runtime.episode_from_outcome(task, world, outcome)
    assert ep.status == STATUS_INCOMPLETE and ep.success is False


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
    assert run.result.objective_score == 1.0


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
