"""agentic harness comparison agentic harnesses -- loop / langgraph / crewai seam, registry, and board comparison."""

import pytest

from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import (
    HARNESS_LOOP,
    AgenticTask,
)
from llb.bench import tool_world as tw
from llb.bench.harness.base import loop_harness
from llb.bench.harness.registry import get_harness
from llb.bench.harness import crewai_runtime
from llb.bench.harness import langgraph as lg


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


def fake_crew_runner(task, complete, catalog, world, max_steps):
    """A fake crew: execute calc+write against the world, then answer (proves the adaptation)."""
    transcript = []
    execute = crewai_runtime.make_recording_executor(world, transcript)
    execute("calculator", {"expression": "12 * (3 + 4)"})
    execute("write_file", {"path": "result.txt", "content": "84"})
    return crewai_runtime.CrewOutcome(
        answer="готово", transcript=transcript, n_steps=3, n_tool_calls=len(transcript)
    )


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


# --- real compiled LangGraph harness (only when the [eval] extra is installed) -------------
