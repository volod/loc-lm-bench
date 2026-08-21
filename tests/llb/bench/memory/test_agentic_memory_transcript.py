"""Reviewable contracts for memory-dependent compact-versus-cap tasks."""

from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic.success import check_success
from llb.bench.memory.transcript import (
    build_memory_dependent_tasks,
    build_token_chain_control_tasks,
    memory_dependent_task,
    token_chain_control_task,
)
from llb.bench.tool_world import ToolWorld


def test_memory_task_requires_early_code_and_externally_stored_progress():
    record = memory_dependent_task(2, depth=3, pad_chars=20)
    task = AgenticTask.from_record(record)
    world = ToolWorld.from_setup(task.setup)
    first = world.execute("advance", {"token": "wf-002-0"})
    code = str(task.success[-2]["value"])
    assert code in first
    world.execute("advance", {"token": "wf-002-1"})
    world.execute("advance", {"token": "wf-002-2"})
    assert check_success(task, world, code)


def test_memory_task_refuses_externalized_final_code():
    record = memory_dependent_task(0, depth=3, pad_chars=0)
    task = AgenticTask.from_record(record)
    world = ToolWorld.from_setup(task.setup)
    first = world.execute("advance", {"token": "wf-000-0"})
    code = str(task.success[-2]["value"])
    assert code in first
    world.execute("advance", {"token": "wf-000-1"})
    world.execute("advance", {"token": "wf-000-2"})
    world.execute("db_set", {"key": "memory", "value": code})
    assert not check_success(task, world, code)


def test_memory_task_set_is_deterministic_and_uses_distinct_codes():
    first = build_memory_dependent_tasks(n_tasks=8)
    second = build_memory_dependent_tasks(n_tasks=8)
    assert first == second
    codes = [task["success"][-2]["value"] for task in first]
    assert len(set(codes)) == len(codes)


def test_token_chain_control_puts_answer_only_in_final_observation():
    record = token_chain_control_task(1, depth=4)
    task = AgenticTask.from_record(record)
    code = str(task.success[-1]["value"])
    workflow = task.setup["workflow"]
    assert all(code not in observation for observation in workflow[:-1])
    assert code in workflow[-1]
    assert build_token_chain_control_tasks(n_tasks=2, depth=4) == [
        token_chain_control_task(0, depth=4),
        token_chain_control_task(1, depth=4),
    ]
