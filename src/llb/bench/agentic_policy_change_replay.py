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
"""

import hashlib
from collections.abc import Mapping
from typing import Any, cast

from llb.bench.agentic.context import POLICY_COMPACT, POLICY_OBSERVATION_CAP, ContextPolicy
from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_memory_boundary_probe import oracle_compacting_controller
from llb.bench.agentic_memory_transcript import build_memory_dependent_tasks

# Both arms of every published cap-fitting cell: its number is the delta between them.
AUDITED_POLICIES = (POLICY_OBSERVATION_CAP, POLICY_COMPACT)


def prompt_sequence_digest(prompts: list[str]) -> str:
    """A stable digest of everything one episode sent, so two replays compare in one field.

    Length-prefixed, so two sequences that differ only in where one prompt ends and the next begins
    do not collide.
    """
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(len(prompt).to_bytes(8, "big"))
        digest.update(prompt.encode("utf-8"))
    return digest.hexdigest()


def replay_prompts(
    policy: ContextPolicy, *, task: AgenticTask, max_prompt_chars: int, max_steps: int
) -> list[str]:
    """Every prompt one oracle episode sends under `policy`, in order.

    Recording through the injected `complete` needs no change to the loop: the callable IS the seam
    every model call passes through, controller prompts and summarize calls alike.
    """
    prompts: list[str] = []

    def recording(prompt: str) -> str:
        prompts.append(prompt)
        return oracle_compacting_controller(prompt)

    run_episode(
        task,
        recording,
        max_steps=max_steps,
        policy=policy,
        budget=fixed_budget(max_prompt_chars),
    )
    return prompts


def arm_comparison(
    policy_name: str,
    cell: dict[str, object],
    held: dict[str, object],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, object]:
    """Replay one arm of one cell under both POLICIES and locate the first prompt that differs.

    `baseline` and `candidate` are whole settings maps, so a change that moves several constants is
    audited as the single change it is: one arm plays the configuration the published number was
    measured under, the other plays the configuration the new build ships.
    """
    tasks = [
        AgenticTask.from_record(record)
        for record in build_memory_dependent_tasks(
            n_tasks=int(cast(int, held["n_tasks"])),
            depth=int(cast(int, cell["depth"])),
            pad_chars=int(cast(int, held["pad_chars"])),
        )
    ]
    max_steps = int(cast(int, cell["depth"])) + int(cast(int, held["max_steps_margin"]))
    guard = int(cast(int, cell["max_prompt_chars"]))

    def episodes(settings: Mapping[str, Any]) -> list[list[str]]:
        return [
            replay_prompts(
                _policy(policy_name, cell, held, settings),
                task=task,
                max_prompt_chars=guard,
                max_steps=max_steps,
            )
            for task in tasks
        ]

    before, after = episodes(baseline), episodes(candidate)
    differing = [
        index
        for index, (left, right) in enumerate(zip(before, after))
        if prompt_sequence_digest(left) != prompt_sequence_digest(right)
    ]
    return {
        "policy": policy_name,
        "identical": not differing,
        "n_tasks": len(tasks),
        "n_differing_tasks": len(differing),
        "first_divergent_step": _first_divergent_step(before, after, differing),
        "baseline_digest": prompt_sequence_digest(
            [prompt for episode in before for prompt in episode]
        ),
        "candidate_digest": prompt_sequence_digest(
            [prompt for episode in after for prompt in episode]
        ),
    }


def _first_divergent_step(
    baseline: list[list[str]], candidate: list[list[str]], differing: list[int]
) -> int | None:
    """The earliest 1-based model call at which any task's two replays stop agreeing."""
    steps = []
    for index in differing:
        left, right = baseline[index], candidate[index]
        pairs = list(zip(left, right))
        diverged = next(
            (step for step, (one, other) in enumerate(pairs, start=1) if one != other), None
        )
        # Equal prefixes with different lengths diverge at the first call only one of them made.
        steps.append(diverged if diverged is not None else len(pairs) + 1)
    return min(steps, default=None)


def _policy(
    policy_name: str,
    cell: dict[str, object],
    held: dict[str, object],
    settings: Mapping[str, Any],
) -> ContextPolicy:
    """The cell's own policy with every audited field overridden -- the overrides always win.

    Fields the change does not touch keep the cell's declared geometry where a design states it, and
    the shipped dataclass default otherwise -- which is the same value on both sides of the audit,
    so it can never be the thing that moves a prompt.
    """
    return ContextPolicy(
        name=policy_name,
        **{
            "observation_cap_chars": int(cast(int, held["observation_cap_chars"])),
            "observation_head_share": float(cast(float, held["observation_head_share"])),
            "compact_share": float(cast(float, cell["compact_share"])),
            **settings,
        },
    )
