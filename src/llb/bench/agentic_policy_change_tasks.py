"""Per-study task builders and oracles for the policy-change audit replay.

Cap-fitting studies walk the memory-dependent token chain. The constant sweep, keep-long lane, and
harness-comparison rows were measured on other shapes, so their audit cells must rebuild those
shapes -- otherwise a keep_last_n change still looks free simply because no memory-chain cell runs
that policy.
"""

import json
from collections.abc import Callable
from typing import Any, cast

from llb.bench.agentic.context_summary import is_summary_prompt
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_long_transcript import pipeline_db_task
from llb.bench.agentic_memory_boundary_probe import ORACLE_SUMMARY, oracle_controller
from llb.bench.agentic_memory_transcript import build_memory_dependent_tasks
from llb.bench.tool_world import DB_SET, FINISH, READ_FILE

# Builders a design's held_fixed may name. Cap-fitting designs omit the field and get the memory one.
TASK_BUILDER_MEMORY = "memory_dependent"
TASK_BUILDER_FAT_OBSERVATION = "fat_observation_pipeline"
TASK_BUILDER_LONG_TRANSCRIPT = "long_transcript_pipeline"
TASK_BUILDER_SEED_SHAPED = "seed_shaped"
TASK_BUILDERS: tuple[str, ...] = (
    TASK_BUILDER_MEMORY,
    TASK_BUILDER_FAT_OBSERVATION,
    TASK_BUILDER_LONG_TRANSCRIPT,
    TASK_BUILDER_SEED_SHAPED,
)


def build_replay_tasks(cell: dict[str, object], held: dict[str, object]) -> list[AgenticTask]:
    """Build the task set one cell was measured on, from the study's declared builder."""
    builder = str(held.get("task_builder", TASK_BUILDER_MEMORY))
    if builder not in TASK_BUILDERS:
        raise ValueError(f"unknown audit task builder {builder!r}; choose from {TASK_BUILDERS}")
    n_tasks = int(cast(int, held["n_tasks"]))
    depth = int(cast(int, cell["depth"]))
    pad_chars = int(cast(int, held["pad_chars"]))
    if builder == TASK_BUILDER_MEMORY:
        return [
            AgenticTask.from_record(record)
            for record in build_memory_dependent_tasks(
                n_tasks=n_tasks, depth=depth, pad_chars=pad_chars
            )
        ]
    if builder == TASK_BUILDER_SEED_SHAPED:
        return [
            AgenticTask.from_record(_seed_shaped_task(index, pad_chars=pad_chars))
            for index in range(n_tasks)
        ]
    # Fat-observation and long-transcript lanes share the pipeline-db shape; only pad / depth differ.
    return [
        AgenticTask.from_record(pipeline_db_task(index, depth=depth, pad_chars=pad_chars))
        for index in range(n_tasks)
    ]


def replay_controller_for(task: AgenticTask, held: dict[str, object]) -> Callable[[str], str]:
    """The oracle that plays one task perfectly under the study's builder."""
    builder = str(held.get("task_builder", TASK_BUILDER_MEMORY))
    if builder == TASK_BUILDER_MEMORY:
        return _memory_controller
    if builder == TASK_BUILDER_SEED_SHAPED:
        return _seed_shaped_oracle(task)
    return _pipeline_db_oracle(task)


def _memory_controller(prompt: str) -> str:
    return ORACLE_SUMMARY if is_summary_prompt(prompt) else oracle_controller(prompt)


def _pipeline_db_oracle(task: AgenticTask) -> Callable[[str], str]:
    """Walk read_file / db_set for every planted file, then finish -- the keep-long intended path."""
    files = cast(dict[str, str], task.setup.get("files", {}))
    paths = [f"n{index}.txt" for index in range(len(files))]
    step = 0

    def complete(prompt: str) -> str:
        nonlocal step
        if is_summary_prompt(prompt):
            return ORACLE_SUMMARY
        pair = step // 2
        if pair < len(paths):
            if step % 2 == 0:
                step += 1
                return json.dumps(
                    {"name": READ_FILE, "arguments": {"path": paths[pair]}}, ensure_ascii=False
                )
            token = files[paths[pair]].split("\n", 1)[0]
            step += 1
            return json.dumps(
                {"name": DB_SET, "arguments": {"key": f"k{pair}", "value": token}},
                ensure_ascii=False,
            )
        return json.dumps({"name": FINISH, "arguments": {"answer": "done"}}, ensure_ascii=False)

    return complete


def _seed_shaped_oracle(task: AgenticTask) -> Callable[[str], str]:
    """One read + one db_set for the single planted file, then finish -- harness seed shape."""
    files = cast(dict[str, str], task.setup.get("files", {}))
    path = next(iter(files))
    step = 0

    def complete(prompt: str) -> str:
        nonlocal step
        if is_summary_prompt(prompt):
            return ORACLE_SUMMARY
        if step == 0:
            step = 1
            return json.dumps({"name": READ_FILE, "arguments": {"path": path}}, ensure_ascii=False)
        if step == 1:
            step = 2
            return json.dumps(
                {
                    "name": DB_SET,
                    "arguments": {"key": "config", "value": files[path].split("\n", 1)[0]},
                },
                ensure_ascii=False,
            )
        return json.dumps({"name": FINISH, "arguments": {"answer": "done"}}, ensure_ascii=False)

    return complete


def _seed_shaped_task(index: int, *, pad_chars: int) -> dict[str, Any]:
    """A short file->db task whose observation sits under the shipped 800-char cap.

    Matches the harness-comparison CUDA evidence shape: seed observations were small enough that
    observation_cap was a no-op, so the audit's honest answer on those cells is invariant under a
    cap re-pin.
    """
    payload = f"mode=fast-{index}"
    # Keep the observation under the shipped cap even when a design pads for determinism.
    text = payload if pad_chars <= 0 else f"{payload}\n{'x' * min(pad_chars, 40)}"
    return {
        "id": f"seed-shaped-{index:03d}",
        "prompt": (
            f"Прочитай файл config{index}.txt і збережи його перший рядок у базі під ключем config."
        ),
        "setup": {"files": {f"config{index}.txt": text}},
        "success": [{"kind": "db_equals", "key": "config", "value": payload}],
    }
