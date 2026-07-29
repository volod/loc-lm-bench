"""The per-step prompt budget the agent loop is checked against -- resolved ONCE per run.

Without this the loop sends every prompt it builds and finds out about an over-long one as a
backend error, a truncated prompt, or (worst) a confidently wrong answer read from a shortened
transcript. With it, a step whose prompt cannot fit terminates the episode as `context_overflow`
-- the status already in the shared taxonomy (`llb.eval.common`) that the context-ablation lane
uses for exactly this situation -- so an unusable configuration is a TYPED outcome instead of a
wrong answer.

The arithmetic is not new: `llb.optimize.tuning_space` already resolves a model's usable window
(host planner cap, model window, served `max_model_len`, explicit `context_budget`) and prices a
prompt against it. This module wraps that in the seam the episode loop needs -- a `fits(chars)`
predicate plus the char budget the `compact` policy measures its trigger share against -- and
imports the heavy resolution lazily so the agentic package stays importable (and unit-testable)
without the backend/hardware stack.
"""

from dataclasses import dataclass
from typing import Callable

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec

# A prompt-char budget of 0 means "cannot bound": no model spec, no served cap, no explicit budget.
# Nothing is refused in that state -- an unknown model must never silently declare a prompt
# unusable, the same rule `fits_context_chars` follows.
UNBOUNDED = 0


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """The resolved per-step prompt budget: how many chars fit, and the predicate that decides."""

    max_prompt_chars: int
    fits: Callable[[int], bool]

    @property
    def bounded(self) -> bool:
        return self.max_prompt_chars > UNBOUNDED

    def compaction_trigger_chars(self, share: float) -> int:
        """The prompt size at which a `compact` policy folds its older steps into a summary.

        A share of the resolved budget rather than a constant, so the same policy compacts late on
        a 32k window and early on a 4k one instead of being tuned per host.
        """
        return int(self.max_prompt_chars * share) if self.bounded else UNBOUNDED


def prompt_tokens(prompt_chars: int) -> int:
    """Project prompt CHARS onto the token estimate every window check in the repo uses.

    The loop measures chars (it has no tokenizer for the served model); `CHARS_PER_TOKEN` is the
    one measured UA conversion the budget arithmetic already rests on, so a reported prompt-token
    column and the guard that refused a prompt are on the same scale.
    """
    from llb.optimize.tuning_space import CHARS_PER_TOKEN

    return int(prompt_chars / CHARS_PER_TOKEN)


def unbounded_budget() -> ContextBudget:
    """The no-guard budget: every prompt fits. The default for callers with no resolved model."""
    return ContextBudget(max_prompt_chars=UNBOUNDED, fits=lambda _chars: True)


def fixed_budget(max_prompt_chars: int) -> ContextBudget:
    """A budget stated directly in prompt chars (the CI seam, and an explicit operator override)."""
    if max_prompt_chars <= UNBOUNDED:
        return unbounded_budget()
    return ContextBudget(
        max_prompt_chars=max_prompt_chars,
        fits=lambda chars: chars <= max_prompt_chars,
    )


def resolve_context_budget(
    config: RunConfig,
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int | None = None,
    ram_mib: int | None = None,
) -> ContextBudget:
    """Resolve the served model's usable prompt budget once, for the whole run.

    `fits` is `fits_context_chars` itself, so the guard and the reported budget can never drift
    apart: `max_prompt_chars` is the largest prompt that same predicate accepts.
    """
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.eval.context_ablation.sources import resolve_model_spec
    from llb.optimize.tuning_space import (
        CHARS_PER_TOKEN,
        PROMPT_HEADROOM_TOKENS,
        effective_max_context,
        fits_context_chars,
    )

    spec = (
        model_spec if model_spec is not None else resolve_model_spec(config.model, config.backend)
    )
    vram = vram_mib if vram_mib is not None else max_vram_mb(detect_gpus())
    ram = ram_mib if ram_mib is not None else detect_ram_mb()

    def fits(prompt_chars: int) -> bool:
        return fits_context_chars(config, spec, vram, ram, prompt_chars)

    window = effective_max_context(config, spec, vram, ram) if spec is not None else UNBOUNDED
    if config.context_budget:
        window = min(window, config.context_budget) if window else config.context_budget
    usable_tokens = window - PROMPT_HEADROOM_TOKENS - config.max_tokens if window else UNBOUNDED
    max_prompt_chars = max(UNBOUNDED, int(usable_tokens * CHARS_PER_TOKEN))
    return ContextBudget(max_prompt_chars=max_prompt_chars, fits=fits)
