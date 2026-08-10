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
from typing import Any

from llb.bench.agentic import context_policy
from llb.bench.agentic.context_aggregate import format_aggregate_header
from llb.bench.agentic.controller_channel import ControllerChannel, ControllerFeedback

# --- policy names (the ranked row labels) --------------------------------------------------

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
# One transcript entry: the tool name, its arguments, and the observation it returned.
TranscriptEntry = tuple[str, dict[str, Any], str]
LOOP_FEEDBACK = "__loop_feedback__"


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
    # What the summarizer was OFFERED versus what its input cap let through. The elided span is the
    # evidence the summary was never shown, so a completion drop under a tighter cap is readable
    # rather than inferred. `summary_fold_input_chars` is the per-fold breakdown of
    # `summary_input_chars`: each compaction appends the transcript it offered, so a multi-fold
    # episode keeps the elision inequality about ONE fold rather than about their sum.
    summary_input_chars: int = 0
    summary_input_elided_chars: int = 0
    n_trimmed_summary_inputs: int = 0
    summary_fold_input_chars: list[int] = field(default_factory=list)

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
        self,
        policy: context_policy.ContextPolicy,
        name: str,
        arguments: dict[str, Any],
        observation: str,
    ) -> None:
        """Append one executed tool call.

        Observation size is counted BEFORE any policy trim -- the point of the column is what the
        world actually returned, so `observation_cap` and `full` stay comparable on it.
        """
        entry: TranscriptEntry = (name, arguments, observation)
        self.entries.append(entry)
        self.executed.append(entry)
        self.telemetry.observation_bytes += len(observation.encode("utf-8"))
        if policy.name in context_policy.TRIMMING_POLICIES:
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
        _policy: context_policy.ContextPolicy,
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
    head_share: float = context_policy.OBSERVATION_HEAD_SHARE,
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


def policy_history_lines(policy: context_policy.ContextPolicy, state: ContextState) -> list[str]:
    """The history lines this policy puts in the next prompt, markers included.

    `full` renders every entry verbatim -- byte-identical to the pre-policy loop, which is what
    lets the baseline row reproduce the recorded agentic rows exactly. `compact` trims live
    observations the same way `observation_cap` does so a fat search hit does not re-blow the
    prompt every step, and when the summary already carries aggregate hit facts a finish cue
    steers the model away from another search/compact cycle.
    """
    entries = state.entries
    dropped = state.n_dropped
    if policy.name == context_policy.POLICY_KEEP_LAST_N and len(entries) > policy.keep_last_n:
        dropped += len(entries) - policy.keep_last_n
        entries = entries[-policy.keep_last_n :] if policy.keep_last_n > 0 else []
    if policy.name in context_policy.TRIMMING_POLICIES:
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
