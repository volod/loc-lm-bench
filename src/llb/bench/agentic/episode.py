"""The controller->execute->controller cycle: build each step prompt, drive one task to an
`Episode` in the deterministic sandbox, resolve/run harnesses, and score a batch of episodes.

`run_episode` is the pure `loop` harness; the LangGraph and CrewAI harnesses (in `llb.bench.harness`)
reuse `build_agent_prompt` + `check_success` to produce the SAME canonical `Episode`.
"""

import json
from typing import Any

from llb.bench.agentic.model import (
    DEFAULT_MAX_STEPS,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    AgenticTask,
    Episode,
    Harness,
    _ScoredAgenticEpisodes,
)
from llb.bench.agentic.success import check_success
from llb.bench.common import LLMComplete, mean
from llb.bench.tool_world import FINISH, ToolWorld, tool_catalog
from llb.core.contracts import AgenticCaseRow, ToolDef
from llb.prompts.registry import render_text
from llb.scoring.leaderboard import bootstrap_mean_ci
from llb.scoring.tool_calls import parse_tool_call


def build_agent_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    transcript: list[tuple[str, dict[str, Any], str]],
) -> str:
    """The next-step prompt: available tools, the task, and the running observation transcript."""
    tools_json = json.dumps(list(catalog.values()), ensure_ascii=False, indent=2)
    history = "\n".join(
        f"- {name}({json.dumps(args, ensure_ascii=False)}) -> {obs}"
        for name, args, obs in transcript
    )
    history_block = (
        render_text("bench.agentic.history_block", {"history": history}) if transcript else ""
    )
    return render_text(
        "bench.agentic.agent_step",
        {
            "tools_json": tools_json,
            "task_prompt": task.prompt,
            "history_block": history_block,
        },
    )


def run_episode(
    task: AgenticTask,
    complete: LLMComplete,
    *,
    catalog: dict[str, ToolDef] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Episode:
    """Drive one task to completion (or the step budget) in the deterministic sandbox.

    This is the pure `loop` harness: the controller->execute->controller cycle with no agent
    framework. `catalog` is injectable so every harness shares ONE tool catalog; it defaults to
    the canonical `tool_catalog()` (so existing callers are unchanged)."""
    world = ToolWorld.from_setup(task.setup)
    catalog = catalog if catalog is not None else tool_catalog()
    transcript: list[tuple[str, dict[str, Any], str]] = []
    answer = ""
    finished = False
    n_tool_calls = 0
    steps = 0
    for steps in range(1, max_steps + 1):
        raw = complete(build_agent_prompt(task, catalog, transcript))
        call = parse_tool_call(raw)
        if call is None:  # the model answered in prose -> treat as the final answer
            answer = raw.strip()
            finished = True
            break
        if call.name == FINISH:
            answer = str(call.arguments.get("answer", ""))
            finished = True
            break
        observation = world.execute(call.name, call.arguments)
        n_tool_calls += 1
        transcript.append((call.name, call.arguments, observation))
    success = check_success(task, world, answer)
    return Episode(
        success=success,
        status=STATUS_COMPLETED if finished else STATUS_INCOMPLETE,
        n_steps=steps,
        n_tool_calls=n_tool_calls,
        answer=answer,
        world=world,
        transcript=transcript,
    )


def _row(task: AgenticTask, episode: Episode) -> AgenticCaseRow:
    return {
        "item_id": task.id,
        "status": episode.status,
        "success": 1.0 if episode.success else 0.0,
        "objective_score": 1.0 if episode.success else 0.0,
        "n_steps": episode.n_steps,
        "n_tool_calls": episode.n_tool_calls,
        "answer_preview": (episode.answer or "")[:280],
    }


def _resolve_harness(harness_name: str, harness: Harness | None) -> Harness:
    if harness is not None:
        return harness
    from llb.bench.harness.registry import get_harness

    return get_harness(harness_name)


def _run_episodes(
    tasks: list[AgenticTask],
    complete: LLMComplete,
    harness: Harness,
    max_steps: int,
) -> list[Episode]:
    catalog = tool_catalog()
    return [harness(task, complete, catalog, max_steps=max_steps) for task in tasks]


def _score_episodes(tasks: list[AgenticTask], episodes: list[Episode]) -> _ScoredAgenticEpisodes:
    rows = [_row(task, episode) for task, episode in zip(tasks, episodes)]
    case_success = [1.0 if episode.success else 0.0 for episode in episodes]
    reliability = sum(1 for episode in episodes if episode.status == STATUS_COMPLETED) / len(
        episodes
    )
    return _ScoredAgenticEpisodes(
        rows=rows,
        case_success=case_success,
        reliability=reliability,
        completion_ci=bootstrap_mean_ci(case_success),
        mean_steps=mean([episode.n_steps for episode in episodes]),
        mean_tool_calls=mean([episode.n_tool_calls for episode in episodes]),
    )
