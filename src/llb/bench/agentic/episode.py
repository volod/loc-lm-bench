"""The controller->execute->controller cycle: build each step prompt, drive one task to an
`Episode` in the deterministic sandbox, resolve/run harnesses, and score a batch of episodes.

`run_episode` is the pure `loop` harness; the LangGraph and CrewAI harnesses (in `llb.bench.harness`)
reuse `build_agent_prompt` + `check_success` to produce the SAME canonical `Episode`.
"""

import json

from llb.bench.agentic.context import (
    POLICY_COMPACT,
    ContextPolicy,
    ContextState,
    TranscriptEntry,
    compact_state,
    format_entry,
    policy_history_lines,
    summarize_entries,
)
from llb.bench.agentic.context_budget import ContextBudget, prompt_tokens, unbounded_budget
from llb.bench.agentic.model import (
    DEFAULT_MAX_STEPS,
    STATUS_COMPLETED,
    STATUS_CONTEXT_OVERFLOW,
    STATUS_INCOMPLETE,
    AgenticTask,
    Episode,
    Harness,
    _ScoredAgenticEpisodes,
)
from llb.bench.agentic.success import check_success
from llb.bench.common import LLMComplete, mean
from llb.bench.tool_world import FINISH, ToolWorld, tool_catalog
from llb.core.contracts.benchmarks import AgenticCaseRow, ToolDef
from llb.prompts.registry import render_text
from llb.scoring.leaderboard import bootstrap_mean_ci
from llb.scoring.tool_calls import parse_tool_call


def build_agent_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    transcript: list[TranscriptEntry],
) -> str:
    """The next-step prompt: available tools, the task, and the running observation transcript."""
    return build_agent_prompt_lines(task, catalog, [format_entry(entry) for entry in transcript])


def build_agent_prompt_lines(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    history_lines: list[str],
) -> str:
    """The next-step prompt from ALREADY-RENDERED history lines.

    The policy seam: a context policy decides which lines (and which markers) the step sees, and
    this assembles the identical prompt scaffold around them. `full` passes every entry through
    verbatim, so its prompt is byte-identical to the pre-policy loop's.
    """
    tools_json = json.dumps(list(catalog.values()), ensure_ascii=False, indent=2)
    history_block = (
        render_text("bench.agentic.history_block", {"history": "\n".join(history_lines)})
        if history_lines
        else ""
    )
    return render_text(
        "bench.agentic.agent_step",
        {
            "tools_json": tools_json,
            "task_prompt": task.prompt,
            "history_block": history_block,
        },
    )


def step_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    policy: ContextPolicy,
    state: ContextState,
    budget: ContextBudget,
    complete: LLMComplete,
) -> str:
    """This step's prompt under `policy`, compacting first when the prompt crosses the trigger.

    At most ONE compaction per step: if the compacted prompt still does not fit, the guard --
    not another round of summarizing -- is what ends the episode. The summarize call is itself
    capped at the trigger size, because its input is the very transcript that just blew the step
    prompt and an over-long summarize call would come back silently truncated.
    """
    prompt = build_agent_prompt_lines(task, catalog, policy_history_lines(policy, state))
    if policy.name != POLICY_COMPACT:
        return prompt
    trigger = budget.compaction_trigger_chars(policy.compact_share)
    if trigger <= 0 or len(prompt) <= trigger:
        return prompt
    summarize = lambda older: summarize_entries(  # noqa: E731
        complete,
        older,
        trigger,
        prior_summary=state.summary,
        telemetry=state.telemetry,
    )
    if not compact_state(policy, state, summarize):
        return prompt
    return build_agent_prompt_lines(task, catalog, policy_history_lines(policy, state))


def run_episode(
    task: AgenticTask,
    complete: LLMComplete,
    *,
    catalog: dict[str, ToolDef] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    policy: ContextPolicy | None = None,
    budget: ContextBudget | None = None,
) -> Episode:
    """Drive one task to completion (or the step budget) in the deterministic sandbox.

    This is the pure `loop` harness: the controller->execute->controller cycle with no agent
    framework. `catalog` is injectable so every harness shares ONE tool catalog; it defaults to
    the canonical `tool_catalog()` (so existing callers are unchanged). `policy` selects the
    context-management policy (default `full`: the whole transcript, today's behavior) and
    `budget` the per-step prompt guard (default unbounded: nothing is refused)."""
    world = ToolWorld.from_setup(task.setup)
    catalog = catalog if catalog is not None else tool_catalog()
    policy = policy if policy is not None else ContextPolicy()
    budget = budget if budget is not None else unbounded_budget()
    state = ContextState()
    answer = ""
    status = STATUS_INCOMPLETE
    n_tool_calls = 0
    steps = 0
    for steps in range(1, max_steps + 1):
        prompt = step_prompt(task, catalog, policy, state, budget, complete)
        state.telemetry.prompt_chars.append(len(prompt))
        if not budget.fits(len(prompt)):
            # The prompt cannot fit the resolved window: end as a TYPED overflow rather than
            # sending it and scoring whatever comes back as the model's answer.
            steps -= 1
            status = STATUS_CONTEXT_OVERFLOW
            break
        state.telemetry.model_input_prompt_chars += len(prompt)
        raw = complete(prompt)
        call = parse_tool_call(raw)
        if call is None:  # the model answered in prose -> treat as the final answer
            answer = raw.strip()
            status = STATUS_COMPLETED
            break
        if call.name == FINISH:
            answer = str(call.arguments.get("answer", ""))
            status = STATUS_COMPLETED
            break
        observation = world.execute(call.name, call.arguments)
        n_tool_calls += 1
        state.record(policy, call.name, call.arguments, observation)
    success = check_success(task, world, answer)
    return Episode(
        success=success,
        status=status,
        n_steps=steps,
        n_tool_calls=n_tool_calls,
        answer=answer,
        world=world,
        transcript=state.executed,
        telemetry=state.telemetry,
        context_policy_supported=True,
    )


def _row(task: AgenticTask, episode: Episode) -> AgenticCaseRow:
    telemetry = episode.telemetry
    row: AgenticCaseRow = {
        "item_id": task.id,
        "status": episode.status,
        "success": 1.0 if episode.success else 0.0,
        "objective_score": 1.0 if episode.success else 0.0,
        "n_steps": episode.n_steps,
        "n_tool_calls": episode.n_tool_calls,
        "answer_preview": (episode.answer or "")[:280],
    }
    if telemetry.prompt_chars:
        # Context accounting rides ALONGSIDE the headline, never in it. A refused prompt is counted
        # here too: the step whose size ended the episode is the one worth seeing.
        row["max_prompt_tokens"] = prompt_tokens(telemetry.max_prompt_chars)
        row["total_prompt_tokens"] = prompt_tokens(telemetry.total_prompt_chars)
        row["total_model_input_tokens"] = prompt_tokens(telemetry.model_input_prompt_chars)
        row["compaction_prompt_tokens"] = prompt_tokens(telemetry.compaction_prompt_chars)
        row["n_model_calls"] = episode.n_steps + telemetry.n_compactions
        row["observation_bytes"] = telemetry.observation_bytes
        row["n_compactions"] = telemetry.n_compactions
        row["n_trimmed_observations"] = telemetry.n_trimmed_observations
    return row


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
    policy: ContextPolicy | None = None,
    budget: ContextBudget | None = None,
) -> list[Episode]:
    catalog = tool_catalog()
    return [
        harness(
            task,
            complete,
            catalog,
            max_steps=max_steps,
            policy=policy,
            budget=budget,
        )
        for task in tasks
    ]


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
