"""Batch execution and objective scoring for canonical agentic episodes."""

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.loop_policy import LoopPolicy
from llb.backends.context_budget import ContextBudget, prompt_tokens
from llb.bench.agentic.model import (
    STATUS_COMPLETED,
    AgenticTask,
    Episode,
    Harness,
    _ScoredAgenticEpisodes,
)
from llb.bench.common import LLMComplete, mean
from llb.bench.tool_world import tool_catalog
from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.scoring.leaderboard import bootstrap_mean_ci


def _row(task: AgenticTask, episode: Episode) -> AgenticCaseRow:
    telemetry = episode.telemetry
    n_model_calls = episode.n_model_calls or episode.n_steps + telemetry.n_compactions
    row: AgenticCaseRow = {
        "item_id": task.id,
        "status": episode.status,
        "success": 1.0 if episode.success else 0.0,
        "objective_score": 1.0 if episode.success else 0.0,
        "n_steps": episode.n_steps,
        "n_tool_calls": episode.n_tool_calls,
        "n_model_calls": n_model_calls,
        "n_malformed_calls": episode.n_malformed_calls,
        "malformed_call_rate": (
            episode.n_malformed_calls / max(episode.n_steps + episode.n_repair_attempts, 1)
        ),
        "n_repair_attempts": episode.n_repair_attempts,
        "n_repeated_calls": episode.n_repeated_calls,
        "n_repeated_noops": episode.n_repeated_noops,
        "repeat_feedback_redirected": episode.repeat_feedback_redirected,
        "elapsed_s": round(episode.elapsed_s, 6),
        "answer_preview": (episode.answer or "")[:280],
    }
    if task.family is not None:
        row["task_family"] = task.family
    if telemetry.prompt_chars:
        row["max_prompt_tokens"] = prompt_tokens(telemetry.max_prompt_chars)
        row["total_prompt_tokens"] = prompt_tokens(telemetry.total_prompt_chars)
        row["total_model_input_tokens"] = prompt_tokens(telemetry.model_input_prompt_chars)
        row["compaction_prompt_tokens"] = prompt_tokens(telemetry.compaction_prompt_chars)
        row["n_model_calls"] = n_model_calls
        row["observation_bytes"] = telemetry.observation_bytes
        row["n_compactions"] = telemetry.n_compactions
        row["n_trimmed_observations"] = telemetry.n_trimmed_observations
        row["summary_input_chars"] = telemetry.summary_input_chars
        row["summary_input_elided_chars"] = telemetry.summary_input_elided_chars
        row["n_trimmed_summary_inputs"] = telemetry.n_trimmed_summary_inputs
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
    loop_policy: LoopPolicy | None = None,
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
            loop_policy=loop_policy,
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
