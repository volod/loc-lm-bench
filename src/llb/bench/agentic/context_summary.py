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
SUMMARY_TRIM_HEAD_TAIL = "head_tail"
SUMMARY_TRIM_PER_ENTRY_HEAD = "per_entry_head"
SUMMARY_TRIM_STRATEGIES = (SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD)
ENTRY_ELISION = "[...entry elided...]"

Summarize = Callable[[list[TranscriptEntry]], str]


def summarize_entries(
    complete: Any,
    entries: list[TranscriptEntry],
    transcript_cap_chars: int = 0,
    *,
    prior_summary: str = "",
    telemetry: ContextTelemetry | None = None,
    trim_strategy: str = SUMMARY_TRIM_HEAD_TAIL,
) -> str:
    """Ask the model for a bounded running summary of older steps."""
    offered = format_summary_transcript(entries, prior_summary=prior_summary)
    transcript, elided = _bounded_summary_transcript(
        entries,
        offered,
        transcript_cap_chars,
        prior_summary=prior_summary,
        trim_strategy=trim_strategy,
    )
    prompt = render_text(COMPACT_SUMMARY_TEMPLATE, {"transcript": transcript})
    if telemetry is not None:
        telemetry.compaction_prompt_chars += len(prompt)
        telemetry.model_input_prompt_chars += len(prompt)
        telemetry.summary_input_chars += len(offered)
        telemetry.summary_fold_input_chars.append(len(offered))
        telemetry.summary_input_elided_chars += len(offered) - transcript_cap_chars if elided else 0
        telemetry.n_trimmed_summary_inputs += 1 if elided else 0
    return (complete(prompt) or "").strip()


def format_summary_transcript(entries: list[TranscriptEntry], *, prior_summary: str = "") -> str:
    """Render the exact transcript offered to the summarizer before its input cap."""
    return "\n".join(_summary_transcript_parts(entries, prior_summary=prior_summary))


def _summary_transcript_parts(
    entries: list[TranscriptEntry], *, prior_summary: str = ""
) -> list[str]:
    """One independently budgetable part per prior summary or transcript entry."""
    enriched = [
        (name, arguments, with_aggregate_header(observation))
        for name, arguments, observation in entries
    ]
    transcript_parts = (
        [f"- [попередній підсумок: {prior_summary}]"] if prior_summary.strip() else []
    )
    transcript_parts.extend(format_entry(entry) for entry in enriched)
    return transcript_parts


def _bounded_summary_transcript(
    entries: list[TranscriptEntry],
    offered: str,
    transcript_cap_chars: int,
    *,
    prior_summary: str,
    trim_strategy: str,
) -> tuple[str, bool]:
    """Apply the shipped whole-transcript trim or the evidence-only entry-aware prototype."""
    if trim_strategy == SUMMARY_TRIM_HEAD_TAIL:
        return trim_observation(offered, transcript_cap_chars, aggregate_safe=False)
    if trim_strategy != SUMMARY_TRIM_PER_ENTRY_HEAD:
        raise ValueError(f"unknown summary trim strategy: {trim_strategy!r}")
    if transcript_cap_chars <= 0 or len(offered) <= transcript_cap_chars:
        return offered, False
    parts = _summary_transcript_parts(entries, prior_summary=prior_summary)
    marker = ELISION.format(dropped=len(offered) - transcript_cap_chars)
    target_chars = transcript_cap_chars + len(marker)
    return _per_entry_head(parts, target_chars), True


def _per_entry_head(parts: list[str], target_chars: int) -> str:
    """Share one fixed byte budget across entries while keeping each entry's leading facts."""
    separators = max(0, len(parts) - 1)
    available = max(0, target_chars - separators)
    base, remainder = divmod(available, len(parts))
    trimmed = [
        _entry_head(part, base + (1 if index < remainder else 0))
        for index, part in enumerate(parts)
    ]
    return "\n".join(trimmed)


def _entry_head(part: str, budget: int) -> str:
    if len(part) <= budget:
        return part
    if budget <= len(ENTRY_ELISION):
        return part[:budget]
    return part[: budget - len(ENTRY_ELISION)] + ENTRY_ELISION


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
            (
                fold_memory_markers(older, prior_summary=state.summary)
                if state.preserve_memory_markers
                else ""
            ),
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
