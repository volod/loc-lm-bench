"""Context-management POLICIES for the agent episode loop -- how a step's prompt spends the window.

The loop rebuilds its prompt from the whole transcript on every step, so one large observation
grows every later prompt for the rest of the episode. That is a policy choice nobody made; this
module makes it a POLICY ROW, exactly as `llb.bench.chain_context_policy` does for question chains:

  - ``full``            -- the whole transcript, verbatim (today's behavior, the baseline row);
  - ``observation_cap`` -- every observation trimmed to a char budget, HEAD and TAIL kept with an
    explicit elision marker so the model can tell it was trimmed rather than truncated silently;
    when trimmed, a machine-computed aggregate header (hit count, total length, matched doc ids)
    is prepended so a count question stays answerable after a middle-of-list loss;
  - ``keep_last_n``     -- only the last N steps survive; the dropped ones are announced as a
    marker line so a missing step is visible instead of looking like it never happened;
  - ``compact``         -- a model-written running summary replaces the older steps once the prompt
    crosses a share of the usable window (the agent-loop counterpart of the chain lane's
    ``summary``). Live steps also get the observation-cap trim (with aggregate headers), and when
    the summary already carries hit-count facts a finish cue tells the model to call ``finish``
    rather than search again -- without those, compact burns the step budget on repeated
    compactions and never answers count tasks.

Everything here is pure and unit-testable over a fake ``complete``: the policy assembles a
deterministic prompt from the transcript, and the telemetry it accumulates (prompt chars per step,
observation bytes, trims, compactions) is what makes the overflow observable after the fact.
`context_budget.py` owns the window arithmetic these policies are measured against.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from llb.bench.agentic.context_aggregate import (
    extract_aggregate_facts,
    format_aggregate_header,
    with_aggregate_header,
)
from llb.bench.agentic.controller_channel import ControllerChannel, ControllerFeedback
from llb.prompts.registry import render_text

# --- policy names (the ranked row labels) --------------------------------------------------

POLICY_FULL = "full"
POLICY_OBSERVATION_CAP = "observation_cap"
POLICY_KEEP_LAST_N = "keep_last_n"
POLICY_COMPACT = "compact"
CONTEXT_POLICIES: tuple[str, ...] = (
    POLICY_FULL,
    POLICY_OBSERVATION_CAP,
    POLICY_KEEP_LAST_N,
    POLICY_COMPACT,
)

# --- policy constants ----------------------------------------------------------------------

# `observation_cap`: chars kept per observation. Sized so a normal tool result (a db value, a
# calculator answer, a search hit list) is untouched and only a dumped file/corpus blob is trimmed.
# PINNED at 800 by the 2026-07-30 constant sweep (cap=400 separates worse on completion; cap=1600
# is flat) -- see extended-workflows.md#agent-context-policy-constants.
DEFAULT_OBSERVATION_CAP_CHARS = 800
# Split of the kept budget between the head and the tail of a trimmed observation. A tool result
# usually leads with its shape and ends with its payload, so both ends carry signal. PINNED at 0.6
# by the same sweep (0.5 / 0.7 flat on completion); exposed as `--observation-head-share`.
OBSERVATION_HEAD_SHARE = 0.6
# `keep_last_n`: how many most-recent steps survive in the prompt. Shipped at 3; the constant sweep
# EXPOSES the knob because keep=1 is flat on completion but separates cheaper on prompt tokens on
# both the short fat-observation set and the longer medium-search set (mean_steps ~8-9) -- do not
# silently rewrite the default from a cost-only reading.
DEFAULT_KEEP_LAST_N = 3
# `compact`: the share of the usable prompt budget that triggers a compaction, and how many recent
# steps stay verbatim behind the summary (the model still needs its immediate working state).
# One recent step is the minimum that keeps a compacting loop from repeating its last tool call,
# and it is also the largest keep that still lets compaction fire on a TWO-step transcript -- a
# larger keep means a prompt blown by one oversized observation has nothing left to compact.
DEFAULT_COMPACT_SHARE = 0.5
DEFAULT_COMPACT_KEEP_RECENT = 1

# Prompt-system template ids owned by this lane.
COMPACT_SUMMARY_TEMPLATE = "bench.agentic.compact_summary"

_ELISION = "[...обрізано {dropped} символів...]"
# A step missing from the prompt is announced, never silently absent: `keep_last_n` says how many
# steps went away with nothing standing in for them, `compact` says how many the summary covers.
_DROPPED_MARKER = "- [опущено попередніх кроків: {dropped}]"
_SUMMARY_MARKER = "- [підсумок попередніх кроків ({dropped}): {summary}]"
# When compaction has already folded search-hit aggregates into the summary, a count task has the
# answer in the prompt -- but without an explicit cue the model keeps searching and burning the
# step budget on another compaction. Name the known hit count so finish is the obvious next call.
_FINISH_CUE = (
    "- [підказка: hits={hits} вже в підсумку -- виклич finish з цією відповіддю, не шукай знову]"
)
_AGGREGATE_HITS = re.compile(r"\[агрегат: hits=(\d+)\b")
_MEMORY_MARKER = re.compile(r"\[memory: [^\]\n]+\]")
_FINAL_CODE_MEMORY = re.compile(r"\[memory: final_code=([^\]\s]+)\]")
_WORKFLOW_COMPLETE = "[workflow complete]"
_MEMORY_FINISH_CUE = (
    '- [підказка: workflow complete; виклич finish з answer="{code}", не викликай advance]'
)

# Policies that trim live observations (and stamp `n_trimmed_observations`).
_TRIMMING_POLICIES = frozenset({POLICY_OBSERVATION_CAP, POLICY_COMPACT})

# One transcript entry: the tool name, its arguments, and the observation it returned.
TranscriptEntry = tuple[str, dict[str, Any], str]
Summarize = Callable[[list[TranscriptEntry]], str]
LOOP_FEEDBACK = "__loop_feedback__"


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """One context-management policy plus the constants it is parameterized by."""

    name: str = POLICY_FULL
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS
    observation_head_share: float = OBSERVATION_HEAD_SHARE
    keep_last_n: int = DEFAULT_KEEP_LAST_N
    compact_share: float = DEFAULT_COMPACT_SHARE
    compact_keep_recent: int = DEFAULT_COMPACT_KEEP_RECENT

    def __post_init__(self) -> None:
        if self.name not in CONTEXT_POLICIES:
            raise ValueError(
                f"unknown context policy: {self.name!r}; choose from {CONTEXT_POLICIES}"
            )
        if not 0.0 < self.observation_head_share < 1.0:
            raise ValueError(
                f"observation_head_share must be in (0, 1), got {self.observation_head_share}"
            )


@dataclass(slots=True)
class ContextTelemetry:
    """Per-episode context accounting -- the numbers that make an overflow observable."""

    prompt_chars: list[int] = field(default_factory=list)
    model_input_prompt_chars: int = 0
    compaction_prompt_chars: int = 0
    observation_bytes: int = 0
    n_trimmed_observations: int = 0
    n_compactions: int = 0
    n_repair_prompts: int = 0

    @property
    def max_prompt_chars(self) -> int:
        return max(self.prompt_chars) if self.prompt_chars else 0

    @property
    def total_prompt_chars(self) -> int:
        return sum(self.prompt_chars)


@dataclass(slots=True)
class ContextState:
    """The transcript the policy manages, plus its running summary and telemetry.

    `entries` is the policy's LIVE VIEW -- what the next prompt may draw on, which `compact` prunes
    as it folds older steps into `summary`. `executed` is the episode's full record of what the
    agent actually did, which the trajectory judge and the persisted transcript read: a policy
    changes what the model SEES, never what the run reports the agent did.
    """

    entries: list[TranscriptEntry] = field(default_factory=list)
    executed: list[TranscriptEntry] = field(default_factory=list)
    summary: str = ""
    n_dropped: int = 0
    controller_feedback: list[ControllerFeedback] = field(default_factory=list)
    telemetry: ContextTelemetry = field(default_factory=ContextTelemetry)

    def record(
        self, policy: "ContextPolicy", name: str, arguments: dict[str, Any], observation: str
    ) -> None:
        """Append one executed tool call.

        Observation size is counted BEFORE any policy trim -- the point of the column is what the
        world actually returned, so `observation_cap` and `full` stay comparable on it.
        """
        entry: TranscriptEntry = (name, arguments, observation)
        self.entries.append(entry)
        self.executed.append(entry)
        self.telemetry.observation_bytes += len(observation.encode("utf-8"))
        if policy.name in _TRIMMING_POLICIES:
            _, trimmed = trim_observation(
                observation,
                policy.observation_cap_chars,
                head_share=policy.observation_head_share,
            )
            self.telemetry.n_trimmed_observations += 1 if trimmed else 0

    def record_feedback(self, message: str) -> None:
        """Put controller feedback in the live prompt without claiming a tool executed."""
        self.entries.append((LOOP_FEEDBACK, {}, message))

    def record_channel_feedback(
        self,
        _policy: "ContextPolicy",
        name: str,
        arguments: dict[str, Any],
        message: str,
        channel: ControllerChannel,
    ) -> None:
        """Record a suppressed call while carrying its notice as a typed chat message."""
        entry: TranscriptEntry = (name, arguments, message)
        self.executed.append(entry)
        self.telemetry.observation_bytes += len(message.encode("utf-8"))
        self.controller_feedback.append(ControllerFeedback(message, channel))


# --- observation trimming ------------------------------------------------------------------


def trim_observation(
    observation: str,
    cap_chars: int,
    *,
    head_share: float = OBSERVATION_HEAD_SHARE,
    aggregate_safe: bool = True,
) -> tuple[str, bool]:
    """Trim to `cap_chars`, keeping the head and the tail around an explicit elision marker.

    Returns `(text, trimmed)`. The marker names how many chars went missing, so a model reading a
    trimmed observation can tell it is looking at a fragment rather than a short tool result.
    When `aggregate_safe` and the observation is trimmed, a machine-computed header of hit count /
    total length / matched doc ids is PREPENDED (outside the cap) so a count question stays
    answerable after a positional middle-of-list loss. `head_share` is the fraction of the kept
    budget given to the leading span (the rest goes to the tail).
    """
    if cap_chars <= 0 or len(observation) <= cap_chars:
        return observation, False
    if not 0.0 < head_share < 1.0:
        raise ValueError(f"head_share must be in (0, 1), got {head_share}")
    head_len = max(1, int(cap_chars * head_share))
    tail_len = max(0, cap_chars - head_len)
    dropped = len(observation) - head_len - tail_len
    marker = _ELISION.format(dropped=dropped)
    tail = observation[-tail_len:] if tail_len else ""
    body = f"{observation[:head_len]}{marker}{tail}"
    if not aggregate_safe:
        return body, True
    return f"{format_aggregate_header(observation)}\n{body}", True


# --- transcript rendering ------------------------------------------------------------------


def format_entry(entry: TranscriptEntry) -> str:
    """The one canonical transcript line: `- tool({args}) -> observation`."""
    name, arguments, observation = entry
    if name == LOOP_FEEDBACK:
        return f"- {observation}"
    return f"- {name}({json.dumps(arguments, ensure_ascii=False)}) -> {observation}"


def summary_hit_count(summary: str) -> int | None:
    """First machine-computed hit count embedded in a compacted summary, if any."""
    match = _AGGREGATE_HITS.search(summary)
    return int(match.group(1)) if match else None


def policy_history_lines(policy: ContextPolicy, state: ContextState) -> list[str]:
    """The history lines this policy puts in the next prompt, markers included.

    `full` renders every entry verbatim -- byte-identical to the pre-policy loop, which is what
    lets the baseline row reproduce the recorded agentic rows exactly. `compact` trims live
    observations the same way `observation_cap` does so a fat search hit does not re-blow the
    prompt every step, and when the summary already carries aggregate hit facts a finish cue
    steers the model away from another search/compact cycle.
    """
    entries = state.entries
    dropped = state.n_dropped
    if policy.name == POLICY_KEEP_LAST_N and len(entries) > policy.keep_last_n:
        dropped += len(entries) - policy.keep_last_n
        entries = entries[-policy.keep_last_n :] if policy.keep_last_n > 0 else []
    if policy.name in _TRIMMING_POLICIES:
        entries = [
            (
                name,
                arguments,
                trim_observation(
                    observation,
                    policy.observation_cap_chars,
                    head_share=policy.observation_head_share,
                )[0],
            )
            for name, arguments, observation in entries
        ]
    lines = [format_entry(entry) for entry in entries]
    if state.summary:
        # The summary STANDS IN for the folded steps, so it carries their count itself -- a
        # separate "dropped" line beside it would read as a second, uncovered loss.
        out = [_SUMMARY_MARKER.format(dropped=dropped, summary=state.summary)]
        hits = summary_hit_count(state.summary)
        if hits is not None:
            out.append(_FINISH_CUE.format(hits=hits))
        code = _FINAL_CODE_MEMORY.search(state.summary)
        if code is not None and any(
            _WORKFLOW_COMPLETE in observation for _, _, observation in entries
        ):
            out.append(_MEMORY_FINISH_CUE.format(code=code.group(1)))
        return out + lines
    return ([_DROPPED_MARKER.format(dropped=dropped)] if dropped else []) + lines


# --- compaction ----------------------------------------------------------------------------


def summarize_entries(
    complete: Any,
    entries: list[TranscriptEntry],
    transcript_cap_chars: int = 0,
    *,
    prior_summary: str = "",
    telemetry: ContextTelemetry | None = None,
) -> str:
    """Ask the model for a running summary of the older steps (the `compact` policy's memory).

    The summarize call is a MODEL CALL like any other, so it is subject to the same window. Its
    input is the transcript that just blew the step prompt, which means an unbounded summarizer is
    the one call in the loop guaranteed to overflow -- and it would overflow into a silently
    truncated summary that the policy then trusts for the rest of the episode. `transcript_cap_chars`
    trims that input the same head-and-tail way an observation is trimmed (0 = no cap, for an
    unresolvable window where nothing is refused anyway).

    Each entry's observation is enriched with an aggregate header BEFORE the joined transcript is
    trimmed, so a free-text summary is still asked to preserve hit counts / doc ids that a
    positional middle-of-list elision would otherwise destroy.
    """
    enriched = [
        (name, arguments, with_aggregate_header(observation))
        for name, arguments, observation in entries
    ]
    transcript_parts = (
        [f"- [попередній підсумок: {prior_summary}]"] if prior_summary.strip() else []
    )
    transcript_parts.extend(format_entry(entry) for entry in enriched)
    # Outer trim is not aggregate-safe: the per-entry headers already carry the facts.
    transcript, _ = trim_observation(
        "\n".join(transcript_parts),
        transcript_cap_chars,
        aggregate_safe=False,
    )
    prompt = render_text(COMPACT_SUMMARY_TEMPLATE, {"transcript": transcript})
    if telemetry is not None:
        telemetry.compaction_prompt_chars += len(prompt)
        telemetry.model_input_prompt_chars += len(prompt)
    return (complete(prompt) or "").strip()


def fold_aggregate_headers(
    entries: list[TranscriptEntry],
    *,
    prior_summary: str = "",
) -> str:
    """Machine aggregate headers for search observations being folded into a summary.

    The compaction prompt already carries these headers, but a free-text summary still drops
    hit counts. Prepending the same machine facts to the summary text makes count survival
    independent of the summarizer.
    """
    headers = re.findall(r"\[агрегат: [^\]]+\]", prior_summary)
    for _name, _arguments, observation in entries:
        facts = extract_aggregate_facts(observation)
        if facts["is_search_hits"]:
            headers.append(format_aggregate_header(observation))
    # De-dupe while preserving order (repeated identical searches collapse to one fact line).
    seen: set[str] = set()
    unique: list[str] = []
    for header in headers:
        if header not in seen:
            seen.add(header)
            unique.append(header)
    return " | ".join(unique)


def fold_memory_markers(
    entries: list[TranscriptEntry],
    *,
    prior_summary: str = "",
) -> str:
    """Preserve typed semantic-memory facts independently of free-text summarization."""
    markers = _MEMORY_MARKER.findall(prior_summary)
    for _name, _arguments, observation in entries:
        markers.extend(_MEMORY_MARKER.findall(observation))
    return " | ".join(dict.fromkeys(markers))


def compact_state(policy: ContextPolicy, state: ContextState, summarize: Summarize) -> bool:
    """Fold the older entries into the running summary in place; True when anything was folded.

    The most recent `compact_keep_recent` steps stay verbatim: they are the agent's working state,
    and summarizing them away is how a compacting loop starts repeating its last tool call. When
    the prompt was blown by that most recent observation there ARE no older steps to fold, and
    keeping it verbatim degenerates the policy into `full`; summarizing the whole transcript is
    then the only thing compaction can do, and it beats sending a prompt the guard will refuse.
    """
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
        # An empty summary would DROP those steps with nothing standing in for them, which is a
        # silent context loss rather than a compaction. Leave the transcript alone instead.
        return False
    state.summary = summary
    state.entries = state.entries[len(older) :]
    state.n_dropped += len(older)
    state.telemetry.n_compactions += 1
    return True
