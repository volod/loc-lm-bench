"""Prompt assembly and context compaction for one agent episode step."""

import json

from llb.bench.agentic.context import (
    ContextState,
    TranscriptEntry,
    format_entry,
    policy_history_lines,
)
from llb.bench.agentic.context_budget import ContextBudget
from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    SUMMARY_INPUT_CAP_TRIGGER,
    ContextPolicy,
)
from llb.bench.agentic.context_summary import (
    compact_state,
    summarize_entries,
    summary_prompt_overhead_chars,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.common import LLMComplete
from llb.core.contracts.benchmarks import ToolDef
from llb.prompts.registry import render_text


def build_agent_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    transcript: list[TranscriptEntry],
) -> str:
    """Build the next-step prompt from a raw transcript."""
    return build_agent_prompt_lines(task, catalog, [format_entry(entry) for entry in transcript])


def build_agent_prompt_lines(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    history_lines: list[str],
) -> str:
    """Build the next-step prompt from policy-rendered history lines."""
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


def summary_input_cap_chars(policy: ContextPolicy, budget: ContextBudget) -> int:
    """Resolve the summary transcript cap from policy and context budget."""
    if policy.summary_input_cap == SUMMARY_INPUT_CAP_TRIGGER:
        return budget.compaction_trigger_chars(policy.compact_share)
    return budget.summary_input_cap_chars(summary_prompt_overhead_chars())


def step_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    policy: ContextPolicy,
    state: ContextState,
    budget: ContextBudget,
    complete: LLMComplete,
) -> str:
    """Build one policy prompt, compacting once when it crosses the trigger."""
    prompt = build_agent_prompt_lines(task, catalog, policy_history_lines(policy, state))
    if policy.name != POLICY_COMPACT:
        return prompt
    trigger_share = 1.0 if state.summary else policy.compact_share
    trigger = budget.compaction_trigger_chars(trigger_share)
    if trigger <= 0 or len(prompt) <= trigger:
        return prompt

    def summarize(older: list[TranscriptEntry]) -> str:
        return summarize_entries(
            complete,
            older,
            summary_input_cap_chars(policy, budget),
            prior_summary=state.summary,
            telemetry=state.telemetry,
        )

    if not compact_state(policy, state, summarize):
        return prompt
    return build_agent_prompt_lines(task, catalog, policy_history_lines(policy, state))
