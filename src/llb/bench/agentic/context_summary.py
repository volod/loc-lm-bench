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
ENTRY_ELISION = "[...entry elided...]"

Summarize = Callable[[list[TranscriptEntry]], str]


def summarize_entries(
    complete: Any,
    entries: list[TranscriptEntry],
    transcript_cap_chars: int = 0,
    *,
    prior_summary: str = "",
    telemetry: ContextTelemetry | None = None,
    trim_strategy: str = context_policy.DEFAULT_SUMMARY_TRIM_STRATEGY,
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
    """Apply the configured whole-transcript trim or the entry-aware one.

    Both spend the SAME byte budget; they differ only in which bytes of the folded transcript
    survive it (`llb.bench.agentic.context_policy.SUMMARY_TRIM_STRATEGIES`).
    """
    if trim_strategy == context_policy.SUMMARY_TRIM_HEAD_TAIL:
        return trim_observation(offered, transcript_cap_chars, aggregate_safe=False)
    if trim_strategy != context_policy.SUMMARY_TRIM_PER_ENTRY_HEAD:
        raise ValueError(
            f"unknown summary trim strategy: {trim_strategy!r}; "
            f"choose from {context_policy.SUMMARY_TRIM_STRATEGIES}"
        )
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


def summary_offered_chars(prompt: str) -> int:
    """How much transcript one summarize prompt is offering, template overhead removed.

    The counterpart of `ContextTelemetry.summary_fold_input_chars` read from the OTHER side: the
    telemetry field records the offered span as the policy folds it, and this recovers the same
    span from the prompt a summarizer is handed. A replayed summarizer needs it because how much
    it writes depends on how much it was shown -- the control's single fold covers the whole
    transcript and a deeper cell's folds cover a few entries each, so a fold length replayed
    without the span it was measured at is a length measured against the wrong offer. The two
    agree exactly whenever the summary-input cap elides nothing, and below it when it does, which
    is the direction that matters: the span the summarizer SAW is what its output length answers.
    """
    return max(0, len(prompt) - len(_empty_summary_prompt()))


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
    # Measured BEFORE the typed facts below are prepended: the fold-length probe replays a
    # summarizer, so what it needs is the span the model itself wrote.
    model_summary_chars = len(summary)
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
    state.telemetry.summary_output_chars.append(model_summary_chars)
    # The step this fold lands on: every completed step has appended its prompt size already, and
    # the fold happens while the NEXT one is being built.
    state.telemetry.summary_fold_steps.append(len(state.telemetry.prompt_chars) + 1)
    state.summary = summary
    state.entries = state.entries[len(older) :]
    state.n_dropped += len(older)
    state.telemetry.n_compactions += 1
    return True
