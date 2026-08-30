"""Replaying one published cell under two policy values and comparing what it sent, byte for byte.

This is the mechanism the policy-change audit rests on. A context policy is a PURE FUNCTION of the
deterministic tool world: fix the geometry and the controller, and the exact sequence of prompts an
episode sends is decided before any model runs. So the invariance test needs no interpretation --
replay, record every prompt, compare.

Two properties make a replay a statement about a REAL run:

  - the summarize call is answered with a FIXED summary, so the replay is deterministic. That only
    ever hides downstream divergence, never invents it: if the summarize prompts are identical then
    a temperature-0 model returns the same summary, so the later controller prompts are identical
    too. "All prompts identical under the replay" therefore implies "all prompts identical under the
    served model" -- the direction the invariance claim needs.
  - both arms of a published cell are replayed (`observation_cap` and `compact`), because a cell's
    published number is a compact-minus-cap delta and a change that moves either arm moves it.

A replay side is a WHOLE policy, never one overridden field. A commit that re-pins two constants
moves them together, so auditing each on its own would compare "pinned cap + shipped keep" against
"shipped cap + shipped keep" -- neither of which is the configuration the published cells were
measured under, and neither of which is the configuration the new build ships.

A replay records the prompt the guard REFUSED as well as the ones it sent, and compares it the same
way -- byte for byte. Recording through `complete` sees only what reached a model, and the loop's
last act before an overflow is to build a prompt, price it, and end the episode WITHOUT sending it
(`budget.fits` in `run_episode`). Two arms that overflow at the same step therefore send
byte-identical prefixes whatever the refused prompts held, so a change that moved the very prompt
that ended the run would read as invariant.

The refused text needs its own seam, because no other one can see it: `run_episode` takes an
`on_refused_prompt` observer and hands it exactly the prompt it refused. A size would not do. The
audited fields include one that moves bytes without moving length -- `observation_head_share`
re-splits a trimmed observation head-and-tail at a fixed cap -- so a refusal compared on its char
count alone is blind to precisely the field whose whole effect is where the bytes went.

The priced SIZE is kept beside the text and comes from the telemetry the loop already stamps
(`prompt_chars`, the refused prompt last). The two are not redundant: under a controller channel the
guard prices a serialized chat transcript, so the size includes a serialization the prompt text does
not show. The audit replays through `complete`, where they agree exactly.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

from llb.bench.agentic.context_policy import (
    DEFAULT_COMPACT_SHARE,
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.policy_change.replay_episode import (
    ReplayedEpisode,
    prompt_sequence_digest,
    replay_digest,
    replay_episode,
    replay_sequence_digest,
)
from llb.bench.policy_change.tasks import build_replay_tasks, replay_controller_for
from llb.bench.common import LLMComplete

# Both arms of every published cap-fitting cell: its number is the delta between them.
AUDITED_POLICIES = (POLICY_OBSERVATION_CAP, POLICY_COMPACT)
# Policy fields a design may hold fixed and that the replay must honour when the change leaves them.
HELD_POLICY_FIELDS = (
    "observation_cap_chars",
    "observation_head_share",
    "keep_last_n",
    "compact_keep_recent",
    "summary_input_cap",
    "summary_trim_strategy",
)


def arm_comparison(
    policy_name: str,
    cell: dict[str, object],
    held: dict[str, object],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    pinned: Mapping[str, Any] | None = None,
    controller: Callable[[AgenticTask, dict[str, object]], LLMComplete] | None = None,
) -> dict[str, object]:
    """Replay one arm of one cell under both POLICIES and locate the first prompt that differs.

    `baseline` and `candidate` are whole settings maps, so a change that moves several constants is
    audited as the single change it is: one arm plays the configuration the published number was
    measured under, the other plays the configuration the new build ships. `pinned`, when supplied
    (the pin gate always does), feeds every field the change does not move so a `restated` pin on a
    held field cannot leave the design's stale value on the baseline arm.

    `controller` selects WHICH walk of the world the two policies are compared over, defaulting to
    the study's own oracle. An invariance verdict is only ever a statement about the transcript it
    was read on, so a caller that needs the claim to cover imperfect play passes the controller
    that produces one instead of re-deriving the comparison.
    """
    tasks = build_replay_tasks(cell, held)
    max_steps = _max_steps(cell, held)
    guard = int(cast(int, cell["max_prompt_chars"]))
    build_controller = controller if controller is not None else replay_controller_for

    def episodes(settings: Mapping[str, Any]) -> list[ReplayedEpisode]:
        return [
            replay_episode(
                _policy(policy_name, cell, held, settings, pinned=pinned),
                task=task,
                max_prompt_chars=guard,
                max_steps=max_steps,
                complete=build_controller(task, held),
            )
            for task in tasks
        ]

    before, after = episodes(baseline), episodes(candidate)
    pairs = list(zip(before, after))
    differing = [
        index
        for index, (left, right) in enumerate(pairs)
        if replay_digest(left) != replay_digest(right)
    ]
    sent_identical = all(
        prompt_sequence_digest(left.prompts) == prompt_sequence_digest(right.prompts)
        for left, right in pairs
    )
    return {
        "policy": policy_name,
        "identical": not differing,
        # What the model saw, ignoring the refusal -- so a divergence can be named as one the
        # episode SENT or one it only built.
        "sent_identical": sent_identical,
        "refused_prompt_moved": any(
            (left.status, left.refused_prompt) != (right.status, right.refused_prompt)
            for left, right in pairs
        ),
        # A refusal the two sides price the SAME and still write differently -- the head share's
        # whole effect, and the case a size comparison could not see.
        "refused_prompt_moved_bytes_only": any(
            left.refused_prompt != right.refused_prompt
            and left.refused_prompt_chars == right.refused_prompt_chars
            for left, right in pairs
        ),
        "n_tasks": len(tasks),
        # How many episodes ended on a prompt the guard refused, per side. Zero on both is what
        # makes a cell's verdict a statement about sent prompts alone.
        "baseline_refused_tasks": sum(record.refused for record in before),
        "candidate_refused_tasks": sum(record.refused for record in after),
        "n_differing_tasks": len(differing),
        "first_divergent_step": _first_divergent_step(before, after, differing),
        "baseline_digest": replay_sequence_digest(before),
        "candidate_digest": replay_sequence_digest(after),
    }


def _first_divergent_step(
    baseline: list[ReplayedEpisode], candidate: list[ReplayedEpisode], differing: list[int]
) -> int | None:
    """The earliest 1-based model call at which any task's two replays stop agreeing.

    A pair whose sent prompts agree but whose refusals do not diverges at the call neither replay
    made: the prompt the guard refused sits one past the last recorded one, which is the same
    position an equal prefix with different lengths already reports.
    """
    steps = []
    for index in differing:
        left, right = baseline[index].prompts, candidate[index].prompts
        pairs = list(zip(left, right))
        diverged = next(
            (step for step, (one, other) in enumerate(pairs, start=1) if one != other), None
        )
        # Equal prefixes with different lengths diverge at the first call only one of them made,
        # and equal sequences that differ only in the refusal diverge at the refused prompt.
        steps.append(diverged if diverged is not None else len(pairs) + 1)
    return min(steps, default=None)


def _policy(
    policy_name: str,
    cell: dict[str, object],
    held: dict[str, object],
    settings: Mapping[str, Any],
    *,
    pinned: Mapping[str, Any] | None = None,
) -> ContextPolicy:
    """The cell's own policy with every audited field overridden -- the overrides always win.

    Fields the change does not touch keep the cell's declared geometry where a design states it, and
    the shipped dataclass default otherwise. When the caller supplies `pinned` (the pin gate always
    does), those untouched fields come from the pins instead -- so a `restated` pin on a held field
    cannot silently put the design's stale value on the baseline arm. A hand-run CLI audit that has
    no pins keeps the design / default fallback. The cell's own `compact_share` stays the cell's
    geometry even when a pin names the held share: a collapse sweep must not flatten to one value.
    """
    share = cell.get("compact_share", held.get("compact_share", DEFAULT_COMPACT_SHARE))
    values: dict[str, Any] = {
        "compact_share": float(cast(float, share)),
    }
    for name in HELD_POLICY_FIELDS:
        if name in held:
            values[name] = held[name]
    if pinned is not None:
        for name, value in pinned.items():
            if name in settings or name == "compact_share":
                continue
            values[name] = value
    values.update(settings)
    return ContextPolicy(name=policy_name, **values)


def _max_steps(cell: dict[str, object], held: dict[str, object]) -> int:
    """Step budget for one cell: an explicit held max_steps, else depth plus the study's margin."""
    if "max_steps" in held:
        return int(cast(int, held["max_steps"]))
    return int(cast(int, cell["depth"])) + int(cast(int, held["max_steps_margin"]))
