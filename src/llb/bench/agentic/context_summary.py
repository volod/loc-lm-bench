"""Bounded summarization and compaction for agent transcript state."""

import re
from functools import cache
from typing import Any, Callable

from llb.bench.agentic import context_policy
from llb.bench.agentic.context import (
    ContextState,
    ContextTelemetry,
    TranscriptEntry,
    format_entry,
    trim_observation,
)
from llb.bench.agentic.context_aggregate import (
    extract_aggregate_facts,
    format_aggregate_header,
    with_aggregate_header,
)
from llb.prompts.registry import render_text

COMPACT_SUMMARY_TEMPLATE = "bench.agentic.compact_summary"
SUMMARY_PROMPT_PREFIX_CHARS = 40
ELISION = "[...обрізано {dropped} символів...]"
ELISION_DROPPED_DIGITS = 12
MEMORY_MARKER = re.compile(r"\[memory: [^\]\n]+\]")

Summarize = Callable[[list[TranscriptEntry]], str]


def summarize_entries(
    complete: Any,
    entries: list[TranscriptEntry],
    transcript_cap_chars: int = 0,
    *,
    prior_summary: str = "",
    telemetry: ContextTelemetry | None = None,
) -> str:
    """Ask the model for a bounded running summary of older steps."""
    enriched = [
        (name, arguments, with_aggregate_header(observation))
        for name, arguments, observation in entries
    ]
    transcript_parts = (
        [f"- [попередній підсумок: {prior_summary}]"] if prior_summary.strip() else []
    )
    transcript_parts.extend(format_entry(entry) for entry in enriched)
    offered = "\n".join(transcript_parts)
    transcript, elided = trim_observation(offered, transcript_cap_chars, aggregate_safe=False)
    prompt = render_text(COMPACT_SUMMARY_TEMPLATE, {"transcript": transcript})
    if telemetry is not None:
        telemetry.compaction_prompt_chars += len(prompt)
        telemetry.model_input_prompt_chars += len(prompt)
        telemetry.summary_input_chars += len(offered)
        telemetry.summary_input_elided_chars += len(offered) - transcript_cap_chars if elided else 0
        telemetry.n_trimmed_summary_inputs += 1 if elided else 0
    return (complete(prompt) or "").strip()


def is_summary_prompt(prompt: str) -> bool:
    """Whether a prompt is the compact policy's summarize call."""
    return prompt.startswith(_empty_summary_prompt()[:SUMMARY_PROMPT_PREFIX_CHARS])


def summary_prompt_overhead_chars() -> int:
    """Characters the summary prompt spends outside its transcript."""
    return len(_empty_summary_prompt()) + len(
        ELISION.format(dropped=10**ELISION_DROPPED_DIGITS - 1)
    )


@cache
def _empty_summary_prompt() -> str:
    return render_text(COMPACT_SUMMARY_TEMPLATE, {"transcript": ""})


def fold_aggregate_headers(
    entries: list[TranscriptEntry],
    *,
    prior_summary: str = "",
) -> str:
    """Preserve machine aggregate headers independently of free-text summarization."""
    headers = re.findall(r"\[агрегат: [^\]]+\]", prior_summary)
    for _name, _arguments, observation in entries:
        facts = extract_aggregate_facts(observation)
        if facts["is_search_hits"]:
            headers.append(format_aggregate_header(observation))
    return " | ".join(dict.fromkeys(headers))


def fold_memory_markers(
    entries: list[TranscriptEntry],
    *,
    prior_summary: str = "",
) -> str:
    """Preserve typed semantic-memory facts independently of free-text summarization."""
    markers = MEMORY_MARKER.findall(prior_summary)
    for _name, _arguments, observation in entries:
        markers.extend(MEMORY_MARKER.findall(observation))
    return " | ".join(dict.fromkeys(markers))


def compact_state(
    policy: context_policy.ContextPolicy,
    state: ContextState,
    summarize: Summarize,
) -> bool:
    """Fold older entries into the running summary; return whether anything was folded."""
    older = state.entries[: max(0, len(state.entries) - policy.compact_keep_recent)]
    if not older:
        older = list(state.entries)
    if not older:
        return False
    summary = (summarize(older) or "").strip()
    facts = " | ".join(
        fact
        for fact in (
            fold_aggregate_headers(older, prior_summary=state.summary),
            fold_memory_markers(older, prior_summary=state.summary),
        )
        if fact
    )
    if facts:
        summary = f"{facts}. {summary}".strip() if summary else facts
    if not summary:
        return False
    state.summary = summary
    state.entries = state.entries[len(older) :]
    state.n_dropped += len(older)
    state.telemetry.n_compactions += 1
    return True
