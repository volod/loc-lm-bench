"""Long-transcript agentic tasks and the keep_last_n-only sweep grid."""

import json

from llb.bench.agentic.context import (
    POLICY_KEEP_LAST_N,
    ContextPolicy,
    ContextState,
    policy_history_lines,
)
from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic.success import check_success
from llb.bench.agentic_context_sweep import (
    AXIS_KEEP,
    keep_long_transcript_grid,
    parse_axes,
    run_constant_sweep,
)
from llb.bench.agentic_long_transcript import (
    build_long_transcript_tasks,
    pipeline_copy_task,
    pipeline_db_task,
    pipeline_sum_task,
)
from llb.bench.agentic.run import load_tasks_file
from llb.bench.tool_world import ToolWorld


def test_pipeline_db_task_succeeds_when_keys_are_written():
    task_dict = pipeline_db_task(0, depth=3, pad_chars=20, values=["a", "b", "c"])
    world = ToolWorld.from_setup(task_dict["setup"])
    for i, val in enumerate(["a", "b", "c"]):
        assert world.execute("read_file", {"path": f"n{i}.txt"}).startswith(val)
        world.execute("db_set", {"key": f"k{i}", "value": val})
    assert check_success(AgenticTask.from_record(task_dict), world, answer="") is True


def test_pipeline_sum_task_succeeds_on_the_intended_path():
    task_dict = pipeline_sum_task(0, depth=3, pad_chars=40, values=[2, 3, 5])
    world = ToolWorld.from_setup(task_dict["setup"])
    for i, n in enumerate([2, 3, 5]):
        world.execute("db_set", {"key": f"k{i}", "value": str(n)})
    assert world.execute("calculator", {"expression": "2+3+5"}) == "10"
    world.execute("write_file", {"path": "sum.txt", "content": "10"})
    assert check_success(AgenticTask.from_record(task_dict), world, answer="") is True


def test_pipeline_copy_task_writes_each_out_file():
    task_dict = pipeline_copy_task(1, depth=2, pad_chars=20)
    world = ToolWorld.from_setup(task_dict["setup"])
    for i in range(2):
        token = world.execute("read_file", {"path": f"src{i}.txt"}).split("\n", 1)[0]
        world.execute("write_file", {"path": f"out{i}.txt", "content": token})
    assert check_success(AgenticTask.from_record(task_dict), world, answer="") is True


def test_medium_observation_search_task_rebinds_count_success():
    task = {
        "id": "search-count-000",
        "prompt": "з'ясуй, у скількох документах корпусу згадується «alpha». Повідом лише число.",
        "setup": {
            "corpus": {
                "a": "alpha here " + ("x" * 500),
                "b": "alpha too " + ("y" * 500),
                "c": "nope " + ("z" * 500),
                "d": "alpha again",
            }
        },
        "success": [{"kind": "answer_contains", "value": "3"}],
    }
    from llb.bench.agentic_long_transcript import medium_observation_search_task

    built = medium_observation_search_task(
        task, max_match_docs=2, max_other_docs=1, max_doc_chars=20
    )
    assert built is not None
    assert built["success"][0]["value"] == "2"  # only 2 matching docs kept
    assert all(len(text) <= 20 for text in built["setup"]["corpus"].values())


def test_build_long_transcript_from_search_tasks_filters_unusable_locate():
    from llb.bench.agentic_long_transcript import build_long_transcript_from_search_tasks

    tasks = [
        {
            "id": "search-locate-000",
            "prompt": "знайди документ, у якому згадується «uniqterm», і повідом його ідентифікатор.",
            "setup": {"corpus": {"only": "uniqterm lives here", "other": "nothing"}},
            "success": [{"kind": "answer_contains", "value": "only"}],
        },
        {
            "id": "search-locate-001",
            "prompt": "знайди документ, у якому згадується «ghost», і повідом його ідентифікатор.",
            # Term only appears past the truncate point -- task must be dropped.
            "setup": {"corpus": {"late": ("pad " * 50) + "ghost"}},
            "success": [{"kind": "answer_contains", "value": "late"}],
        },
    ]
    built = build_long_transcript_from_search_tasks(tasks, max_doc_chars=20)
    assert len(built) == 1
    assert built[0]["id"] == "medium-search-locate-000"


def test_keep_long_transcript_grid_is_keep_only():
    grid = keep_long_transcript_grid()
    assert {s.axis for s in grid} == {AXIS_KEEP}
    assert [s.label for s in grid] == ["keep=1", "keep=2", "keep=3"]
    assert parse_axes("keep_last_n") == (AXIS_KEEP,)


def test_keep_policy_drops_older_steps_on_a_deep_transcript():
    state = ContextState()
    policy = ContextPolicy(name=POLICY_KEEP_LAST_N, keep_last_n=1)
    for i in range(5):
        state.record(policy, "read_file", {"path": f"n{i}.txt"}, f"obs-{i}")
    lines = policy_history_lines(policy, state)
    assert any("опущено попередніх кроків: 4" in line for line in lines)
    assert lines[-1].endswith("obs-4")


def test_run_keep_axis_sweep_over_pipeline_tasks(tmp_path):
    """Fake endpoint walks the intended db path; keep grid pairs under raised max_steps."""
    tasks_path = tmp_path / "long.json"
    raw = build_long_transcript_tasks(n_db=2, n_copy=0, n_sum=0, depth=3, pad_chars=30)
    tasks_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    tasks = load_tasks_file(tasks_path)
    call_state = {"task_i": 0, "step": 0}

    def complete(_prompt: str) -> str:
        task = tasks[call_state["task_i"] % len(tasks)]
        depth = len(task.setup.get("files", {}))
        step = call_state["step"]
        call_state["step"] += 1
        # Alternate read / db_set for each index, then finish.
        pair = step // 2
        if pair < depth:
            if step % 2 == 0:
                return json.dumps(
                    {"name": "read_file", "arguments": {"path": f"n{pair}.txt"}},
                    ensure_ascii=False,
                )
            token = task.setup["files"][f"n{pair}.txt"].split("\n", 1)[0]
            return json.dumps(
                {"name": "db_set", "arguments": {"key": f"k{pair}", "value": token}},
                ensure_ascii=False,
            )
        call_state["task_i"] += 1
        call_state["step"] = 0
        return '{"name":"finish","arguments":{"answer":"done"}}'

    run = run_constant_sweep(
        tasks,
        model="fake",
        backend="fake",
        complete=complete,
        axes=(AXIS_KEEP,),
        max_steps=12,
        budget=fixed_budget(50_000),
        data_dir=tmp_path,
    )
    assert len(run.settings) == 3
    assert run.verdicts[0].axis == AXIS_KEEP
    assert all(s.report.mean_steps > 3 for s in run.settings)
    assert all(s.report.result.objective_score == 1.0 for s in run.settings)
    assert "warning:" not in run.verdicts[0].reason
