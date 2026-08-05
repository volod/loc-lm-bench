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

Live backends can disagree with the declared window (Ollama's default `num_ctx` is 4096 regardless
of a GGUF advertising 131072). Resolution therefore takes the MINIMUM of the declared window and
a probed `served_max_model_len`, and records which one bound the budget.
"""

from dataclasses import dataclass
from typing import Callable

from llb.backends.served_window import (
    BUDGET_SOURCE_DECLARED,
    BUDGET_SOURCE_FIXED,
    BUDGET_SOURCE_SERVED,
    BUDGET_SOURCE_UNBOUNDED,
    HttpGet,
    bind_window,
    probe_config_served_max_model_len,
)
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
    declared_max_model_len: int | None = None
    served_max_model_len: int | None = None
    budget_source: str = BUDGET_SOURCE_UNBOUNDED

    @property
    def bounded(self) -> bool:
        return self.max_prompt_chars > UNBOUNDED

    def compaction_trigger_chars(self, share: float) -> int:
        """The prompt size at which a `compact` policy folds its older steps into a summary.

        A share of the resolved budget rather than a constant, so the same policy compacts late on
        a 32k window and early on a 4k one instead of being tuned per host.
        """
        return int(self.max_prompt_chars * share) if self.bounded else UNBOUNDED

    def summary_input_cap_chars(self, overhead_chars: int) -> int:
        """The largest summarize INPUT that still fits the window once its template is paid for.

        The summarize call is bounded by the SAME window every controller prompt is, so this is the
        widest cap that keeps it from being refused -- and, unlike the compaction trigger, it does
        not move with `compact_share`. A folded transcript below it is summarized whole.
        """
        return max(1, self.max_prompt_chars - overhead_chars) if self.bounded else UNBOUNDED

    def provenance(self) -> dict[str, object]:
        """Manifest fields naming the declared window, the probed window, and which bound."""
        return {
            "declared_max_model_len": self.declared_max_model_len,
            "served_max_model_len": self.served_max_model_len,
            "budget_source": self.budget_source,
        }


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
    return ContextBudget(
        max_prompt_chars=UNBOUNDED,
        fits=lambda _chars: True,
        budget_source=BUDGET_SOURCE_UNBOUNDED,
    )


def fixed_budget(max_prompt_chars: int) -> ContextBudget:
    """A budget stated directly in prompt chars (the CI seam, and an explicit operator override)."""
    if max_prompt_chars <= UNBOUNDED:
        return unbounded_budget()
    return ContextBudget(
        max_prompt_chars=max_prompt_chars,
        fits=lambda chars: chars <= max_prompt_chars,
        budget_source=BUDGET_SOURCE_FIXED,
    )


def _declared_window(
    config: RunConfig,
    model_spec: ModelSpec | None,
    vram_mib: int,
    ram_mib: int,
) -> int:
    """The DECLARED usable window before any live backend probe (0 == cannot bound)."""
    from llb.optimize.tuning_space import effective_max_context

    window = (
        effective_max_context(config, model_spec, vram_mib, ram_mib)
        if model_spec is not None
        else UNBOUNDED
    )
    if config.context_budget:
        window = min(window, config.context_budget) if window else config.context_budget
    return window


def _budget_from_window(
    config: RunConfig,
    window: int,
    *,
    declared_max_model_len: int,
    served_max_model_len: int | None,
    budget_source: str,
) -> ContextBudget:
    from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    usable_tokens = window - PROMPT_HEADROOM_TOKENS - config.max_tokens if window else UNBOUNDED
    max_prompt_chars = max(UNBOUNDED, int(usable_tokens * CHARS_PER_TOKEN))

    def fits(prompt_chars: int) -> bool:
        # Bound against the SAME char budget the report prints, so a served-window bind cannot
        # drift from fits_context_chars' declared-only path when model_spec is missing.
        return max_prompt_chars <= UNBOUNDED or prompt_chars <= max_prompt_chars

    return ContextBudget(
        max_prompt_chars=max_prompt_chars,
        fits=fits,
        declared_max_model_len=declared_max_model_len or None,
        served_max_model_len=served_max_model_len,
        budget_source=budget_source,
    )


def resolve_context_budget(
    config: RunConfig,
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int | None = None,
    ram_mib: int | None = None,
    served_max_model_len: int | None = None,
    probe: bool = False,
    http_get: HttpGet | None = None,
) -> ContextBudget:
    """Resolve the usable prompt budget once, for the whole run.

    When `probe` is true (or `served_max_model_len` is passed), the budget is the MINIMUM of the
    declared window and the probed served window. A probe miss falls back to the declared window
    and records `budget_source=declared` with `served_max_model_len=None`.
    """
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.eval.context_ablation.sources import resolve_model_spec

    spec = (
        model_spec if model_spec is not None else resolve_model_spec(config.model, config.backend)
    )
    vram = vram_mib if vram_mib is not None else max_vram_mb(detect_gpus())
    ram = ram_mib if ram_mib is not None else detect_ram_mb()
    declared = _declared_window(config, spec, vram, ram)
    probed = served_max_model_len
    if probe and probed is None:
        probed = probe_config_served_max_model_len(config, http_get=http_get)
    window, source = bind_window(declared, probed)
    if source == BUDGET_SOURCE_UNBOUNDED:
        return ContextBudget(
            max_prompt_chars=UNBOUNDED,
            fits=lambda _chars: True,
            declared_max_model_len=declared or None,
            served_max_model_len=probed,
            budget_source=BUDGET_SOURCE_UNBOUNDED,
        )
    # bind_window never returns "fixed"; keep declared/served as-is.
    if source not in (BUDGET_SOURCE_DECLARED, BUDGET_SOURCE_SERVED):
        source = BUDGET_SOURCE_DECLARED if declared > 0 else BUDGET_SOURCE_SERVED
    return _budget_from_window(
        config,
        window,
        declared_max_model_len=declared,
        served_max_model_len=probed,
        budget_source=source,
    )
