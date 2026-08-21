"""Replaying ONE episode under one policy, and digesting exactly what it put in front of the guard.

A replay records the prompts an episode SENT through the injected `complete` seam, and the prompt
the guard REFUSED through the loop's own refusal observer -- the refused one never reaches a model,
so nothing downstream of `complete` can see it. Both halves are digested the same length-prefixed
way, so a comparison is byte for byte rather than by size: the audited fields include one
(`observation_head_share`) whose whole effect is WHERE the bytes went, not how many there are.

`agentic_policy_change_replay` compares two arms built out of these.
"""

import hashlib
from dataclasses import dataclass, field

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import STATUS_CONTEXT_OVERFLOW, AgenticTask, Episode
from llb.bench.common import LLMComplete


@dataclass(frozen=True, slots=True)
class ReplayedEpisode:
    """Everything one replayed episode put in front of the guard: sent prompts, and the refused one.

    `refused_prompt` is the text that ended the episode as `context_overflow` and
    `refused_prompt_chars` the size the guard priced it at; both are None for any other terminal
    status -- an episode that finished refused nothing.
    """

    prompts: list[str] = field(default_factory=list)
    status: str = ""
    refused_prompt: str | None = None
    refused_prompt_chars: int | None = None

    @property
    def refused(self) -> bool:
        return self.refused_prompt is not None


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


def replay_digest(record: ReplayedEpisode) -> str:
    """A stable digest of one episode: what it sent, how it ended, and what the guard refused.

    The refused prompt is digested through the same length-prefixed helper the sent ones use, so
    it is compared byte for byte rather than by size.
    """
    digest = hashlib.sha256()
    digest.update(prompt_sequence_digest(record.prompts).encode("ascii"))
    digest.update(f"|{record.status}|{record.refused_prompt_chars}|".encode("utf-8"))
    digest.update(prompt_sequence_digest([record.refused_prompt or ""]).encode("ascii"))
    return digest.hexdigest()


def replay_sequence_digest(records: list[ReplayedEpisode]) -> str:
    """One digest for a whole arm's tasks -- of the per-episode digests, so no boundary collides."""
    return prompt_sequence_digest([replay_digest(record) for record in records])


def replay_episode(
    policy: ContextPolicy,
    *,
    task: AgenticTask,
    max_prompt_chars: int,
    max_steps: int,
    complete: LLMComplete | None = None,
) -> ReplayedEpisode:
    """One oracle episode under `policy`: every prompt it sends, plus the one the guard refuses.

    Recording the SENT prompts through the injected `complete` needs no change to the loop: the
    callable IS the seam every model call passes through, controller prompts and summarize calls
    alike. The refused prompt never reaches that seam, so the loop's own refusal observer supplies
    it -- at most once per episode, since a refusal ends the episode.

    `complete`, when supplied, is the study's own oracle (pipeline / seed-shaped / memory). Cap-
    fitting callers that omit it keep the memory-chain compacting oracle via the task builder.
    """
    from llb.bench.memory.boundary.probe import oracle_compacting_controller

    prompts: list[str] = []
    refused: list[str] = []
    controller = complete if complete is not None else oracle_compacting_controller

    def recording(prompt: str) -> str:
        prompts.append(prompt)
        return controller(prompt)

    episode = run_episode(
        task,
        recording,
        max_steps=max_steps,
        policy=policy,
        budget=fixed_budget(max_prompt_chars),
        on_refused_prompt=refused.append,
    )
    return ReplayedEpisode(
        prompts=prompts,
        status=episode.status,
        refused_prompt=refused[0] if refused else None,
        refused_prompt_chars=_refused_prompt_chars(episode),
    )


def _refused_prompt_chars(episode: Episode) -> int | None:
    """The size of the prompt the guard refused, or None when the episode ended some other way.

    The loop stamps `prompt_chars` for every prompt it PRICES, so the refused one is the last entry
    of an overflowed episode -- the same value `budget.fits` said no to.
    """
    if episode.status != STATUS_CONTEXT_OVERFLOW:
        return None
    priced = episode.telemetry.prompt_chars
    return priced[-1] if priced else None
