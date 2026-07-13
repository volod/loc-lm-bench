"""agentic harness comparison agentic harnesses -- loop / langgraph / crewai seam, registry, and board comparison."""

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
from llb.bench.harness import get_harness, loop_harness
from llb.bench.harness import crewai as crewai_harness
from llb.bench.harness import langgraph as lg
from llb.board.harnesses import harness_comparison, load_agentic_harness_records


def scripted(outputs):
    it = iter(list(outputs))
    return lambda _prompt: next(it)


SUCCESS_SCRIPT = [
    '{"name":"calculator","arguments":{"expression":"12 * (3 + 4)"}}',
    '{"name":"write_file","arguments":{"path":"result.txt","content":"84"}}',
    '{"name":"finish","arguments":{"answer":"готово"}}',
]


def success_task():
    return AgenticTask(
        "t",
        "обчисли і запиши",
        setup={},
        success=[{"kind": "file_equals", "path": "result.txt", "value": "84"}],
    )


# --- registry -----------------------------------------------------------------------------


def test_get_harness_loop_is_loop_harness():
    assert get_harness(HARNESS_LOOP) is loop_harness


def test_get_harness_unknown_raises():
    with pytest.raises(SystemExit):
        get_harness("nope")


def test_loop_harness_matches_run_episode():
    catalog = tw.tool_catalog()
    a = loop_harness(success_task(), scripted(SUCCESS_SCRIPT), catalog)
    b = run_episode(success_task(), scripted(SUCCESS_SCRIPT), catalog=catalog)
    assert (a.success, a.status, a.n_steps, a.n_tool_calls, a.answer) == (
        b.success,
        b.status,
        b.n_steps,
        b.n_tool_calls,
        b.answer,
    )
    assert a.success is True and a.n_tool_calls == 2 and a.n_steps == 3


# --- langgraph harness pure nodes + equivalence (no langgraph install needed) --------------


def test_agent_node_stages_tool_call():
    node = lg.make_agent_node(
        scripted(['{"name":"db_get","arguments":{"key":"k"}}']), tw.tool_catalog()
    )
    update = node({"task": success_task(), "transcript": [], "n_steps": 0})
    assert update["pending_name"] == "db_get" and update["pending_args"] == {"key": "k"}
    assert update["finished"] is False and update["n_steps"] == 1


def test_agent_node_finish_and_prose():
    finish = lg.make_agent_node(scripted(['{"name":"finish","arguments":{"answer":"ok"}}']), {})
    upd = finish({"task": success_task(), "transcript": []})
    assert upd["finished"] is True and upd["answer"] == "ok"
    prose = lg.make_agent_node(scripted(["просто відповідь"]), {})
    upd2 = prose({"task": success_task(), "transcript": []})
    assert upd2["finished"] is True and upd2["answer"] == "просто відповідь"


def test_tool_node_executes_and_records():
    node = lg.make_tool_node()
    world = tw.ToolWorld.from_setup({})
    upd = node(
        {
            "world": world,
            "pending_name": "write_file",
            "pending_args": {"path": "a.txt", "content": "x"},
            "transcript": [],
        }
    )
    assert world.files["a.txt"] == "x"
    assert upd["n_tool_calls"] == 1 and upd["transcript"][0][0] == "write_file"


def test_route_after_agent_and_tool():
    assert lg.route_after_agent({"finished": True}) == lg.ROUTE_END
    assert lg.route_after_agent({"finished": False}) == lg.ROUTE_TOOL
    assert lg.route_after_tool({"n_steps": 1, "max_steps": 3}) == lg.ROUTE_AGENT
    assert lg.route_after_tool({"n_steps": 3, "max_steps": 3}) == lg.ROUTE_END


@pytest.mark.parametrize(
    "script,task,max_steps",
    [
        (SUCCESS_SCRIPT, success_task(), 6),
        # budget exhausted: always calls a tool, never finishes
        (
            ['{"name":"db_get","arguments":{"key":"k"}}'] * 5,
            AgenticTask("t", "p", success=[{"kind": "answer_contains", "value": "x"}]),
            3,
        ),
        # answers in prose on step 1
        (
            ["просто текст"],
            AgenticTask("t", "p", success=[{"kind": "answer_contains", "value": "текст"}]),
            6,
        ),
    ],
)
def test_langgraph_nodes_reproduce_the_loop(script, task, max_steps):
    """The graph's pure nodes + routers must produce the SAME Episode as run_episode."""
    catalog = tw.tool_catalog()
    loop_ep = run_episode(task, scripted(script), catalog=catalog, max_steps=max_steps)
    graph_ep = lg.step_graph_pure(task, scripted(script), catalog, max_steps=max_steps)
    assert (
        graph_ep.success,
        graph_ep.status,
        graph_ep.n_steps,
        graph_ep.n_tool_calls,
        graph_ep.answer,
    ) == (loop_ep.success, loop_ep.status, loop_ep.n_steps, loop_ep.n_tool_calls, loop_ep.answer)
    assert graph_ep.world.files == loop_ep.world.files
    assert graph_ep.transcript == loop_ep.transcript


# --- crewai harness (fake crew, no dependency) ---------------------------------------------


def test_recording_executor_runs_and_records():
    world = tw.ToolWorld.from_setup({"db": {"k": "v"}})
    transcript = []
    execute = crewai_harness.make_recording_executor(world, transcript)
    assert execute("db_get", {"key": "k"}) == "v"
    assert execute("db_set", {"key": "k2", "value": "v2"}) == tw.OBS_OK
    assert world.db["k2"] == "v2"
    assert [name for name, _a, _o in transcript] == ["db_get", "db_set"]


def test_crew_tool_specs_excludes_finish():
    specs = crewai_harness.crew_tool_specs(tw.tool_catalog(), lambda n, a: "")
    names = {s.name for s in specs}
    assert tw.FINISH not in names and tw.WRITE_FILE in names


def fake_crew_runner(task, complete, catalog, world, max_steps):
    """A fake crew: execute calc+write against the world, then answer (proves the adaptation)."""
    transcript = []
    execute = crewai_harness.make_recording_executor(world, transcript)
    execute("calculator", {"expression": "12 * (3 + 4)"})
    execute("write_file", {"path": "result.txt", "content": "84"})
    return crewai_harness.CrewOutcome(
        answer="готово", transcript=transcript, n_steps=3, n_tool_calls=len(transcript)
    )


def test_crewai_harness_with_fake_crew():
    harness = crewai_harness.make_crewai_harness(fake_crew_runner)
    ep = harness(success_task(), scripted([]), tw.tool_catalog(), max_steps=6)
    assert ep.success is True and ep.status == STATUS_COMPLETED
    assert ep.n_tool_calls == 2 and ep.answer == "готово"


def test_episode_from_outcome_incomplete_when_not_finished():
    task = AgenticTask("t", "p", success=[{"kind": "answer_contains", "value": "x"}])
    world = tw.ToolWorld.from_setup({})
    outcome = crewai_harness.CrewOutcome(answer="", finished=False)
    ep = crewai_harness.episode_from_outcome(task, world, outcome)
    assert ep.status == STATUS_INCOMPLETE and ep.success is False


# --- run_agentic records the harness + board comparison ------------------------------------


def two_tasks():
    return [
        AgenticTask(
            "a",
            "calc+write",
            success=[{"kind": "file_equals", "path": "result.txt", "value": "84"}],
        ),
        AgenticTask("b", "db", success=[{"kind": "db_equals", "key": "capital", "value": "Київ"}]),
    ]


def loop_script():
    return scripted(
        [
            '{"name":"calculator","arguments":{"expression":"12 * (3 + 4)"}}',
            '{"name":"write_file","arguments":{"path":"result.txt","content":"84"}}',
            '{"name":"finish","arguments":{"answer":"done"}}',
            '{"name":"db_set","arguments":{"key":"capital","value":"Київ"}}',
            '{"name":"finish","arguments":{"answer":"done"}}',
        ]
    )


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


# --- real compiled LangGraph harness (only when the [eval] extra is installed) -------------


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
