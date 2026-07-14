"""agentic workflows -- tool-world, success checks, episode loop, runner."""

from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import (
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    AgenticTask,
)
from llb.bench.agentic.success import check_assertion, check_success
from llb.bench import tool_world as tw


# --- deterministic tool-world -------------------------------------------------------------


def test_safe_eval_arithmetic():
    assert tw.safe_eval("12 * (3 + 4)") == "84"
    assert tw.safe_eval("10 / 4") == "2.5"
    assert tw.safe_eval("2 ** 10") == "1024"


def test_safe_eval_rejects_code():
    assert tw.safe_eval("__import__('os')") == tw.OBS_CALC_ERROR
    assert tw.safe_eval("len([1,2])") == tw.OBS_CALC_ERROR
    assert tw.safe_eval("1/0") == tw.OBS_CALC_ERROR


def test_tool_world_files_and_db():
    world = tw.ToolWorld.from_setup({"files": {"a.txt": "hi"}})
    assert world.execute(tw.READ_FILE, {"path": "a.txt"}) == "hi"
    assert world.execute(tw.READ_FILE, {"path": "missing"}) == tw.OBS_FILE_NOT_FOUND
    assert world.execute(tw.WRITE_FILE, {"path": "b.txt", "content": "x"}) == tw.OBS_OK
    assert world.files["b.txt"] == "x"
    assert world.execute(tw.DB_SET, {"key": "k", "value": "v"}) == tw.OBS_OK
    assert world.execute(tw.DB_GET, {"key": "k"}) == "v"
    assert world.execute(tw.DB_GET, {"key": "nope"}) == tw.OBS_DB_MISSING


def test_tool_world_search_and_calculator_and_unknown():
    world = tw.ToolWorld.from_setup({"corpus": {"d1": "бюджет зріс на 15%"}})
    assert "d1" in world.execute(tw.SEARCH, {"query": "бюджет"})
    assert world.execute(tw.SEARCH, {"query": "хмарочос"}) == tw.OBS_NO_RESULTS
    assert world.execute(tw.CALCULATOR, {"expression": "2+2"}) == "4"
    assert world.execute("nope", {}) == tw.OBS_UNKNOWN_TOOL


def test_tool_world_bad_args():
    world = tw.ToolWorld()
    assert world.execute(tw.WRITE_FILE, {"path": "a"}) == tw.OBS_BAD_ARGS


# --- success assertions -------------------------------------------------------------------


def test_check_assertions():
    world = tw.ToolWorld(files={"r.txt": "84"}, db={"capital": "Київ"})
    assert check_assertion({"kind": "file_equals", "path": "r.txt", "value": "84"}, world, "")
    assert check_assertion({"kind": "db_equals", "key": "capital", "value": "київ"}, world, "")
    assert check_assertion({"kind": "answer_contains", "value": "15"}, world, "зросло на 15%")
    assert not check_assertion({"kind": "file_equals", "path": "r.txt", "value": "0"}, world, "")


def test_check_success_requires_all_and_nonempty():
    world = tw.ToolWorld(db={"k": "v"})
    task = AgenticTask("t", "p", success=[{"kind": "db_equals", "key": "k", "value": "v"}])
    assert check_success(task, world, "") is True
    empty = AgenticTask("t", "p", success=[])
    assert check_success(empty, world, "") is False  # no assertions never passes


# --- episode loop -------------------------------------------------------------------------


def scripted(outputs):
    it = iter(outputs)
    return lambda _prompt: next(it)


def test_run_episode_success_with_tools():
    task = AgenticTask(
        "t",
        "обчисли і запиши",
        setup={},
        success=[{"kind": "file_equals", "path": "result.txt", "value": "84"}],
    )
    complete = scripted(
        [
            '{"name":"calculator","arguments":{"expression":"12 * (3 + 4)"}}',
            '{"name":"write_file","arguments":{"path":"result.txt","content":"84"}}',
            '{"name":"finish","arguments":{"answer":"готово"}}',
        ]
    )
    ep = run_episode(task, complete)
    assert ep.success is True
    assert ep.status == STATUS_COMPLETED
    assert ep.n_tool_calls == 2 and ep.n_steps == 3


def test_run_episode_budget_exhausted_is_incomplete():
    task = AgenticTask("t", "p", success=[{"kind": "answer_contains", "value": "x"}])
    # always calls a tool, never finishes
    complete = lambda _: '{"name":"db_get","arguments":{"key":"k"}}'  # noqa: E731
    ep = run_episode(task, complete, max_steps=3)
    assert ep.status == STATUS_INCOMPLETE
    assert ep.n_steps == 3 and ep.success is False


# --- runner -------------------------------------------------------------------------------


# --- opt-in gated-judge trajectory quality (agentic benchmark residual) ---------------------------------


def fake_judge(faith=0.8, relevancy=0.6):
    """A judge scorer returning fixed G-Eval signals per record (no DeepEval / endpoint)."""

    def scorer(records, _model):
        return [{"faithfulness": faith, "answer_relevancy": relevancy} for _ in records]

    return scorer
